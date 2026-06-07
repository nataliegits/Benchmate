"""Streamlit UI for Benchmate.

Run with:
    cd ~/Desktop/Benchmate
    source .venv/bin/activate
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

# Make repo imports work whether this is run from repo root or ui/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from co_scientist.tools import available_geneformer_genes, geneformer_neighbors
from ui.notebook_gen import generate_notebook, resolve_to_ensembl, CELL_TYPE_PRESETS
from ui.colab_handoff import handoff, gh_available, GhUnavailable
from co_scientist.llm_config import model_for, _DEFAULT_ROLE_MODELS


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "geneformer"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Benchmate", layout="wide")

# Inter for body text, Source Serif 4 for display — matches the editorial
# research-tool aesthetic (Elicit-style). Loaded from Google Fonts; falls
# back to system sans if blocked.
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif !important;
            font-feature-settings: 'cv11', 'ss01', 'ss03';
            letter-spacing: -0.005em;
        }
        h1, h2, h3 {
            font-family: 'Source Serif 4', Georgia, serif !important;
            letter-spacing: -0.015em;
            font-weight: 600;
        }
        h1 { font-size: 2.4rem !important; }
        h2 { font-size: 1.5rem !important; }
        code, pre, .stCode {
            font-family: 'JetBrains Mono', 'SF Mono', Menlo,
                         monospace !important;
        }
        /* Subtle teal accent on primary buttons */
        .stButton button[kind="primary"] {
            background-color: #0e7490;
            border-color: #0e7490;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #155e75;
            border-color: #155e75;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with a thin accent rule above the title for editorial polish
st.markdown(
    "<div style='border-top: 3px solid #0e7490; width: 56px; "
    "margin-bottom: 8px;'></div>",
    unsafe_allow_html=True,
)
st.title("Benchmate")
st.markdown(
    "<p style='color:#57534e; font-size:1.05rem; margin-top:-8px; "
    "font-family: Inter, sans-serif;'>"
    "An AI co-scientist for biomedical hypothesis generation, grounded "
    "in your own perturbation data.</p>",
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────
# Sidebar: keys, cache, uploads, model routing
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Anthropic API key")
    # CRITICAL: do NOT pre-populate this field from os.environ. Even with
    # type="password" the rendered DOM value is visible to anyone using
    # browser DevTools. Each user enters their own key from scratch; the
    # value lives only in their session.
    if "anthropic_key" not in st.session_state:
        st.session_state.anthropic_key = ""
    user_key = st.text_input(
        "Your Anthropic key",
        type="password",
        placeholder="sk-ant-...",
        key="anthropic_key",
        help="Available at console.anthropic.com. Stored only in your "
             "browser session; never sent to the Benchmate maintainer.",
    )
    if user_key:
        # Only set the env var for this Python process if the user provided
        # a key in this session. Never fall back to a maintainer-set env
        # var that would leak across sessions.
        os.environ["ANTHROPIC_API_KEY"] = user_key
        st.caption("Key set for this session.")
    else:
        # Make sure no leaked key from a previous session lingers
        os.environ.pop("ANTHROPIC_API_KEY", None)
        st.caption("Required for the Run Benchmate tab.")

    st.divider()
    st.subheader("Cached perturbations")
    cached = available_geneformer_genes()
    if cached:
        st.metric("genes", len(cached))
        with st.container(border=True):
            for g in cached:
                st.markdown(f"`{g}`")
    else:
        st.info("No cached perturbations yet.")

    st.divider()
    st.subheader("Upload CSVs")
    uploaded = st.file_uploader(
        "Drop *_stats.csv files here",
        type=["csv"],
        accept_multiple_files=True,
        help="After running the Colab notebook, drag the downloaded "
             "*_stats.csv files into this box.",
    )
    if uploaded:
        for f in uploaded:
            name = f.name
            if not name.endswith("_stats.csv"):
                st.warning(f"Skipping {name} (expected *_stats.csv)")
                continue
            dst = CACHE_DIR / name
            dst.write_bytes(f.read())
            st.success(f"Cached {name}")
        st.rerun()

    st.divider()
    with st.expander("Model routing", expanded=False):
        st.caption(
            "Assign a model to each agent role. The default sends "
            "reasoning-heavy roles to Sonnet and throughput roles to Haiku. "
            "Changes apply to the next Run Benchmate call."
        )
        MODEL_OPTIONS = [
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-opus-4-7",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash",
        ]
        ROLE_PRICE = {
            "anthropic/claude-sonnet-4-6": 3.0,
            "anthropic/claude-haiku-4-5": 1.0,
            "anthropic/claude-opus-4-7": 15.0,
            "openai/gpt-5": 2.5,
            "openai/gpt-5-mini": 0.25,
            "gemini/gemini-2.5-pro": 1.25,
            "gemini/gemini-2.5-flash": 0.30,
        }
        ROLES = ["generation", "reflection", "evolution",
                 "ranking", "meta_review", "supervisor"]
        for role in ROLES:
            current = model_for(role)
            if current not in MODEL_OPTIONS:
                MODEL_OPTIONS.insert(0, current)
            idx = MODEL_OPTIONS.index(current)
            choice = st.selectbox(
                role,
                MODEL_OPTIONS,
                index=idx,
                key=f"model_{role}",
                help=f"~${ROLE_PRICE.get(current, '?')}/1M input tokens",
            )
            if choice != current:
                os.environ[f"BENCHMATE_MODEL_{role.upper()}"] = choice
        st.caption(
            "Generation, Reflection, and Evolution benefit most from a "
            "strong model — they read the Geneformer evidence. The others "
            "can run cheaper without measurable quality loss."
        )

# ────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "New perturbation",
    "Inspect cache",
    "Run Benchmate",
    "Hermes preview",
    "Benchmark",
])

# ── Tab 1 ────────────────────────────────────────────────────
with tab1:
    st.header("Add genes to the perturbation cache")
    st.write(
        "Enter the gene symbols you want to perturb and pick the cell "
        "context. Benchmate generates a Colab notebook with both pre-filled."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        genes_in = st.text_input(
            "Gene symbols (comma-separated)",
            placeholder="e.g. XBP1, ATF4, EIF2AK3",
        )
    with col2:
        preset_name = st.selectbox(
            "Cell context",
            list(CELL_TYPE_PRESETS.keys()),
            index=0,
            help="The cells Geneformer will perturb your genes in. "
                 "Pick where your genes' biology should be readable.",
        )
    st.caption(CELL_TYPE_PRESETS[preset_name]["rationale"])

    if st.button("Generate Colab notebook", type="primary"):
        if not genes_in.strip():
            st.error("Enter at least one gene symbol.")
        else:
            symbols = [g.strip().upper() for g in genes_in.split(",") if g.strip()]
            with st.spinner("Resolving Ensembl IDs..."):
                try:
                    resolved = resolve_to_ensembl(symbols)
                except Exception as e:
                    st.error(f"mygene lookup failed: {e}")
                    st.stop()
            unresolved = [s for s in symbols if s not in resolved]
            if unresolved:
                st.warning(f"Could not resolve: {', '.join(unresolved)}")
            if not resolved:
                st.error("No symbols resolved. Check spelling.")
                st.stop()

            st.success(f"Resolved {len(resolved)} gene(s) for context: {preset_name}")
            st.json(resolved)

            with st.spinner("Generating notebook..."):
                nb_path, _ = generate_notebook(resolved.keys(), preset_name=preset_name)
            st.write(f"Notebook saved: `{nb_path.relative_to(REPO_ROOT)}`")

            if gh_available():
                with st.spinner("Pushing to Gist..."):
                    try:
                        urls = handoff(nb_path, description=f"Benchmate: {', '.join(resolved)}")
                    except Exception as e:
                        urls = None
                        st.warning(f"Gist push failed: {e}. Falling back to download.")
                if urls:
                    st.markdown(f"### [Open in Colab]({urls['colab_url']})")
                    st.caption(f"Gist: {urls['gist_url']}")
            else:
                st.info(
                    "GitHub CLI not available. Download the notebook below, "
                    "open colab.research.google.com, and use "
                    "File → Upload notebook to load it."
                )
                st.download_button(
                    "Download notebook",
                    nb_path.read_bytes(),
                    file_name=nb_path.name,
                    mime="application/x-ipynb+json",
                )

            st.divider()
            st.info(
                "In Colab: Runtime → Change runtime type → T4 GPU, then "
                "Runtime → Run all. When the notebook finishes, the final "
                "cell auto-downloads each *_stats.csv to your browser. "
                "Drag those files into the Upload CSVs panel in the sidebar."
            )

# ── Tab 2 ────────────────────────────────────────────────────
with tab2:
    st.header("Browse the cache")
    if not cached:
        st.info("Cache is empty.")
    else:
        gene = st.selectbox("Pick a cached gene", cached)
        n = st.slider("How many top affected genes", 5, 50, 10)
        r = geneformer_neighbors(gene, top_n=n)
        if "error" in r:
            st.error(r["error"])
        else:
            st.write(f"Top {r['n_results']} affected genes when {gene} was deleted:")
            import pandas as pd
            df = pd.DataFrame(r["affected_genes"])
            st.dataframe(df, use_container_width=True)

# ── Tab 3 ────────────────────────────────────────────────────
with tab3:
    st.header("Run the Co-Scientist loop")
    goal = st.text_area(
        "Research goal",
        height=140,
        value=(
            "Generate testable hypotheses for how TXNDC15 couples ERAD to "
            "mitophagy in human cells. Use SYVN1 and MARCHF6 as known-comparator "
            "ERAD E3 ligases. Prioritise experiments feasible in HEK293 or "
            "HepG2 with standard proteostasis tools (CCCP, thapsigargin, "
            "tunicamycin, cycloheximide, co-IP, immunoblotting)."
        ),
    )
    iterations = st.slider("Max iterations", 4, 16, 8)
    cost_estimate = 0.30 * iterations / 4
    st.caption(f"Estimated Anthropic spend: ~${cost_estimate:.2f} "
               f"({iterations} iterations).")

    if st.button("Run Benchmate", type="primary"):
        st.info("Running. Log lines stream below; the page stays responsive.")
        log_box = st.empty()
        log_lines: list[str] = []
        proc = subprocess.Popen(
            [sys.executable, "run.py", goal, "--max-iterations", str(iterations)],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            log_lines.append(line.rstrip())
            log_box.code("\n".join(log_lines[-120:]))
        proc.wait()
        if proc.returncode == 0:
            st.success("Benchmate finished.")
            state_file = REPO_ROOT / "state.json"
            if state_file.exists():
                # Parse and display top hypotheses as a real Streamlit table
                # so column headers stay visible regardless of log scrollback.
                state_data = json.loads(state_file.read_text())
                hyps = sorted(
                    state_data.get("hypotheses", []),
                    key=lambda h: (h.get("matches_played", 0) > 0,
                                   h.get("elo", 0)),
                    reverse=True,
                )[:5]
                if hyps:
                    import pandas as pd
                    rows = []
                    for h in hyps:
                        played = h.get("matches_played", 0)
                        rows.append({
                            "Elo": round(h.get("elo", 0), 0),
                            "Matches": played,
                            "Gen": h.get("generation", 0),
                            "Ranked": "yes" if played > 0 else "no (untested)",
                            "Statement": h.get("statement", ""),
                        })
                    df = pd.DataFrame(rows)
                    st.subheader(f"Top {len(hyps)} hypotheses")
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Elo": st.column_config.NumberColumn(
                                width="small",
                                help="Tournament rating (starts at 1200). "
                                     "Wins push up, losses pull down."),
                            "Matches": st.column_config.NumberColumn(
                                width="small",
                                help="Number of pairwise tournament rounds "
                                     "this hypothesis has been judged in. "
                                     "Higher = more reliable rating."),
                            "Gen": st.column_config.NumberColumn(
                                width="small",
                                help="Generation: 0 = original Generation "
                                     "output, 1 = first Evolution refinement, "
                                     "2 = second, etc."),
                            "Ranked": st.column_config.TextColumn(
                                width="small",
                                help="'no' means the hypothesis was created "
                                     "by Evolution after the last Ranking "
                                     "round, so its Elo is still the default "
                                     "1200 and isn't a quality signal."),
                            "Statement": st.column_config.TextColumn(width="large"),
                        },
                    )
                    # Per-hypothesis expand for rationale + experiment
                    for i, h in enumerate(hyps, 1):
                        with st.expander(
                            f"#{i} — Elo {round(h.get('elo', 0), 0):.0f} "
                            f"— rationale + experiment"
                        ):
                            st.markdown("**Rationale**")
                            st.markdown(h.get("rationale", ""))
                            st.markdown("**Proposed experiment**")
                            st.markdown(h.get("experiment", ""))
                            if h.get("review_notes"):
                                st.markdown("**Reviewer notes**")
                                for note in h["review_notes"]:
                                    st.markdown(f"- {note}")
                st.download_button(
                    "Download state.json",
                    state_file.read_text(),
                    file_name="state.json",
                )
        else:
            st.error(f"Benchmate exited with code {proc.returncode}.")

# ── Tab 4 ────────────────────────────────────────────────────
with tab4:
    st.header("Hermes preview")
    st.write(
        "Hermes is an autonomous agent (hermes-agent.nousresearch.com) that "
        "runs on a server and accepts messages from Slack, Discord, "
        "Telegram, email, or the CLI. When wired to Benchmate, it calls the "
        "JSON-in / JSON-out functions previewed below."
    )
    st.caption(
        "This tab shows the inputs and outputs of those functions so you "
        "can validate the integration before deploying Hermes. Full "
        "deployment guide: HERMES.md."
    )

    from hermes.benchmate_runner import (
        list_cache as hermes_list_cache,
        gene_neighbors as hermes_neighbors,
        add_perturbation as hermes_add,
    )

    with st.container(border=True):
        st.markdown("**list_cache()** — what genes are in the cache?")
        st.caption("Chat equivalent: \"hermes, what perturbations do we have cached?\"")
        if st.button("Run list_cache", key="hermes_list"):
            result = hermes_list_cache()
            st.code(json.dumps(result, indent=2), language="json")

    with st.container(border=True):
        st.markdown("**gene_neighbors(symbol, top_n)** — top affected genes for a knockout")
        st.caption("Chat equivalent: \"hermes, what does TXNDC15 knockout shift downstream?\"")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            sym = st.selectbox("Gene", cached or ["(no cached genes)"],
                               key="hermes_neighbors_gene")
        with col_b:
            top_n = st.number_input("Top N", 3, 30, 5,
                                    key="hermes_neighbors_top_n")
        if st.button("Run gene_neighbors", key="hermes_neighbors_btn"):
            if cached:
                result = hermes_neighbors(sym, top_n=int(top_n))
                st.code(json.dumps(result, indent=2, default=str),
                        language="json")
            else:
                st.warning("No cached genes — upload some CSVs first.")

    with st.container(border=True):
        st.markdown("**add_perturbation(symbols, cell_context)** — generate a Colab notebook")
        st.caption("Chat equivalent: \"hermes, add FOXP3 to the perturbation queue in plasma cells.\"")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            new_genes = st.text_input("Gene symbols (comma-separated)",
                                      value="FOXP3", key="hermes_add_genes")
        with col_b:
            new_ctx = st.selectbox("Cell context",
                                   list(CELL_TYPE_PRESETS.keys()),
                                   index=2,
                                   key="hermes_add_ctx")
        if st.button("Run add_perturbation", key="hermes_add_btn"):
            syms = [g.strip().upper() for g in new_genes.split(",") if g.strip()]
            result = hermes_add(syms, cell_context=new_ctx)
            st.code(json.dumps(result, indent=2, default=str), language="json")

    with st.container(border=True):
        st.markdown("**run_benchmate(goal, max_iterations)** — execute the full agent loop")
        st.caption("Chat equivalent: \"hermes, run benchmate on this goal: ...\"")
        st.info(
            "This call costs API credits (roughly $0.30–$3 depending on "
            "iterations). Use the Run Benchmate tab for live runs. The "
            "JSON shape Hermes receives is:"
        )
        st.code(json.dumps({
            "iterations_run": 8,
            "n_hypotheses": 24,
            "top_hypotheses": [
                {
                    "elo": 1287.4,
                    "statement": "TXNDC15 is an ER-luminal thioredoxin...",
                    "rationale": "The Geneformer KO signature for...",
                    "experiment": "1) MAM FRACTIONATION WITH POSITIVE...",
                    "matches_played": 6,
                    "generation": 1,
                },
                "... up to 5 hypotheses ranked by Elo ..."
            ],
        }, indent=2), language="json")

    st.divider()
    st.markdown(
        "### Wire Hermes to Slack\n"
        "All four functions also run as CLI commands "
        "(`python -m hermes.benchmate_runner list-cache`). To make them "
        "chat-driven, deploy Hermes on a small VPS and register a skill "
        "that wraps the CLI. The full guide is in "
        "[HERMES.md](https://github.com/nataliegits/Benchmate/blob/main/HERMES.md)."
    )

# ── Tab 5 ────────────────────────────────────────────────────
with tab5:
    st.header("Is the Elo leaderboard trustworthy?")
    st.write(
        "The simulator replays Benchmate's **real** `elo.py` against synthetic "
        "hypotheses whose true quality we control — for free, thousands of "
        "times — so you can see how match budget, K-factor, and judge skill "
        "move ranking accuracy and repeatability before spending API credits."
    )
    st.caption(
        "Reads as: **spearman**→ranking accuracy vs ground truth "
        "(1.0=perfect), **top1**→true best ends #1, **repeat**→two runs pick "
        "the same winner, **churn**→rank movement in the final cycle "
        "(→0 = converged)."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sim_n = st.slider("Hypotheses (n)", 4, 12, 6,
                          help="How many hypotheses are in the simulated tournament.")
    with col_b:
        sim_skill = st.slider("Judge skill", 0.55, 0.90, 0.70, 0.05,
                              help=("P(judge picks the better of two adjacent "
                                    "hypotheses). 0.5 = coin flip; ~0.70 is "
                                    "decent. Measure your real judge with "
                                    "`judge-eval` and feed the number back in."))
    with col_c:
        sim_replicates = st.select_slider(
            "Replicates", options=[50, 100, 200, 300], value=200,
            help="More replicates = tighter error bars but a slower run."
        )

    st.markdown("**Pick a study** — what question do you want the simulator to answer?")
    study = st.radio(
        "Study",
        ["Match budget — how many matches do I need?",
         "K-factor — does the 40/20/10 schedule matter?",
         "Judge quality — how badly does a weak / biased judge hurt?"],
        label_visibility="collapsed",
    )

    if st.button("Run simulator", type="primary"):
        from benchmark.simulate import (evaluate, CURRENT_K, k_const,
                                         k_schedule)

        rows: list[dict] = []
        with st.spinner("Replaying tournaments…"):
            if study.startswith("Match budget"):
                configs = [(1, 6, "1 round (12 matches)"),
                           (2, 6, "2 rounds (24 matches)"),
                           (4, 6, "4 rounds (48 matches)"),
                           (6, 8, "6 rounds, 8/cycle (96)"),
                           (10, 10, "10 rounds, 10/cycle (200)")]
                for cycles, npc, lbl in configs:
                    s = evaluate(lbl, replicates=sim_replicates,
                                 n=sim_n, cycles=cycles, n_per_cycle=npc,
                                 judge_skill=sim_skill)
                    rows.append({
                        "config": s.label,
                        "matches/hyp": round(s.matches_per_hyp, 1),
                        "spearman": round(s.mean_spearman, 2),
                        "± sd": round(s.sd_spearman, 2),
                        "top1": f"{s.top1_accuracy:.0%}",
                        "top3": f"{s.top3_accuracy:.0%}",
                        "repeat": f"{s.repeatability:.0%}",
                        "churn": round(s.final_churn, 2),
                    })
            elif study.startswith("K-factor"):
                configs = [("current 40/20/10", CURRENT_K),
                           ("constant K=32 (FIDE)", k_const(32)),
                           ("constant K=16", k_const(16)),
                           ("gentle 24/16/8",
                            k_schedule([(5, 24), (15, 16)], 8)),
                           ("hot 64/32/16",
                            k_schedule([(5, 64), (15, 32)], 16))]
                for lbl, k_fn in configs:
                    s = evaluate(lbl, replicates=sim_replicates,
                                 n=sim_n, cycles=4, n_per_cycle=6,
                                 judge_skill=sim_skill, k_fn=k_fn)
                    rows.append({
                        "config": s.label,
                        "matches/hyp": round(s.matches_per_hyp, 1),
                        "spearman": round(s.mean_spearman, 2),
                        "± sd": round(s.sd_spearman, 2),
                        "top1": f"{s.top1_accuracy:.0%}",
                        "top3": f"{s.top3_accuracy:.0%}",
                        "repeat": f"{s.repeatability:.0%}",
                        "churn": round(s.final_churn, 2),
                    })
            else:  # judge quality
                configs = [("strong (skill 80%)", 0.80, 0.0),
                           ("decent (skill 70%)", 0.70, 0.0),
                           ("weak (skill 60%)", 0.60, 0.0),
                           ("decent + 15% position bias", 0.70, 0.15)]
                for lbl, skill_v, pbias in configs:
                    s = evaluate(lbl, replicates=sim_replicates,
                                 n=sim_n, cycles=4, n_per_cycle=6,
                                 judge_skill=skill_v, position_bias=pbias)
                    rows.append({
                        "config": s.label,
                        "matches/hyp": round(s.matches_per_hyp, 1),
                        "spearman": round(s.mean_spearman, 2),
                        "± sd": round(s.sd_spearman, 2),
                        "top1": f"{s.top1_accuracy:.0%}",
                        "top3": f"{s.top3_accuracy:.0%}",
                        "repeat": f"{s.repeatability:.0%}",
                        "churn": round(s.final_churn, 2),
                    })

        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        if study.startswith("Match budget"):
            st.info(
                "**Read:** top-3 is reliable around ~8 matches/hyp; a "
                "trustworthy *#1* needs ~12+. Your `state.json` today shows "
                "~2 matches/hyp."
            )
        elif study.startswith("K-factor"):
            st.info(
                "**Read:** at this scale the K-factor barely moves the "
                "metrics. The 40/20/10 schedule, flat K=16, and flat K=32 "
                "are within noise. Match budget is the lever, not K."
            )
        else:
            st.info(
                "**Read:** judge skill is the accuracy ceiling — improving "
                "the judge buys as much as doubling the match budget. The "
                "fair judge (default) neutralises position bias by judging "
                "each pair in both orders."
            )

    st.divider()
    st.markdown("### Live benchmarks (run from the CLI)")
    st.markdown(
        "The simulator is free. The next three benchmarks make real API "
        "calls — keep them on the CLI so you can watch the spend:"
    )
    st.code(
        "# Is the LLM judge any good? Accuracy / position-bias / consistency\n"
        "python -m benchmark.run_benchmark judge-eval --max-pairs 8\n\n"
        "# Rank the ERAD gold set end-to-end and score it vs the tier order\n"
        "python -m benchmark.run_benchmark validate --cycles 6 --n-per-cycle 8\n\n"
        "# Same gold set, fair judge vs naive judge, side by side\n"
        "python -m benchmark.run_benchmark compare",
        language="bash",
    )
    st.caption(
        "Full plan and recommended sequence of work: "
        "[benchmark/BENCHMARKING_PLAN.md]"
        "(https://github.com/nataliegits/Benchmate/blob/main/benchmark/"
        "BENCHMARKING_PLAN.md). Gold set lives in `benchmark/gold_set.py` "
        "(currently ERAD / bortezomib-resistant multiple myeloma)."
    )
