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
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif !important;
            letter-spacing: -0.02em;
            font-weight: 600;
        }
        h1 { font-size: 2.4rem !important; }
        h2 { font-size: 1.5rem !important; }
        code, pre, .stCode {
            font-family: 'JetBrains Mono', 'SF Mono', Menlo,
                         monospace !important;
        }
        /* Monochrome editorial — ink primary buttons */
        .stButton button[kind="primary"] {
            background-color: #111111;
            border-color: #111111;
            color: #ffffff;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #000000;
            border-color: #000000;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with a thin accent rule above the title for editorial polish
st.markdown(
    "<div style='border-top: 3px solid #111111; width: 56px; "
    "margin-bottom: 8px;'></div>",
    unsafe_allow_html=True,
)
st.title("Benchmate")
st.markdown(
    "<p style='color:#555555; font-size:1.05rem; margin-top:-8px; "
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
DEFAULT_GOAL = (
    "Generate testable hypotheses for how TXNDC15 couples ERAD to "
    "mitophagy in human cells. Use SYVN1 and MARCHF6 as known-comparator "
    "ERAD E3 ligases. Prioritise experiments feasible in HEK293 or "
    "HepG2 with standard proteostasis tools (CCCP, thapsigargin, "
    "tunicamycin, cycloheximide, co-IP, immunoblotting)."
)
_PRESET_KEYS = list(CELL_TYPE_PRESETS.keys())
# Defaults for the inputs the "Start here" tab pre-fills in the other tabs.
st.session_state.setdefault("genes_in", "")
st.session_state.setdefault("preset_name", _PRESET_KEYS[0])
st.session_state.setdefault("goal", DEFAULT_GOAL)
st.session_state.setdefault("run_iterations", 8)
st.session_state.setdefault("sh_plan", None)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Start here",
    "New perturbation",
    "Inspect cache",
    "Run Benchmate",
    "Benchmark",
    "Cross-check with other models",
])

# ── Tab 0 — guided flow ──────────────────────────────────────
with tab0:
    st.header("Start here — from question to cross-checked hypotheses")
    st.write(
        "Type a research question. Benchmate reads it and lays out the exact "
        "steps: which genes to perturb, in which cell type, and which models "
        "to cross-check the winning hypotheses against."
    )

    st.session_state.setdefault(
        "sh_question",
        "What ERAD genes drive bortezomib resistance in multiple myeloma?",
    )
    question = st.text_area("Your research question", key="sh_question", height=90)

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not have_key:
        st.info("Add your Anthropic API key in the sidebar to build a plan.")

    if st.button("Build my plan", type="primary", disabled=not have_key):
        from co_scientist.llm import call_json
        presets = list(CELL_TYPE_PRESETS.keys())
        prompt = (
            "You are planning a Benchmate run for a biomedical research "
            "question. Decide:\n"
            "1. perturb_genes: 2-4 human gene SYMBOLS most worth perturbing "
            "to generate evidence for this question.\n"
            f"2. cell_context: choose EXACTLY one string from this list: "
            f"{presets}.\n"
            "3. cell_reason: <=15 words on why that cell context fits.\n"
            "4. iterations: an integer 4-16 for the tournament size.\n"
            "5. cross_check: 2-4 items, each an object with keys target (a "
            "gene or protein symbol the hypotheses will likely center on), "
            "model (EXACTLY one of: AlphaGenome, Boltz, Open Targets, DepMap, "
            "AlphaMissense), and reason (<=10 words). Match the model to the "
            "target: Open Targets/DepMap for gene-disease questions, Boltz for "
            "drug-target binding, AlphaGenome for regulatory variants, "
            "AlphaMissense for coding variants.\n\n"
            f"Question: {question}"
        )
        try:
            with st.spinner("Reading your question…"):
                plan = call_json(prompt, role="generation", max_tokens=900)
            if plan.get("cell_context") not in CELL_TYPE_PRESETS:
                plan["cell_context"] = presets[0]
            st.session_state.sh_plan = plan
        except Exception as e:
            st.error(f"Could not build a plan: {e}")

    plan = st.session_state.get("sh_plan")
    if plan:
        genes = [str(g).upper() for g in plan.get("perturb_genes", []) if str(g).strip()]
        cell = plan.get("cell_context", _PRESET_KEYS[0])
        iters = max(4, min(16, int(plan.get("iterations", 8) or 8)))
        xchecks = plan.get("cross_check", []) or []

        st.divider()
        st.caption("Your path — three steps. Each fills the right tab for you.")

        with st.container(border=True):
            st.markdown("**1 · Generate your evidence**  —  New perturbation tab")
            st.markdown(
                f"Perturb **{', '.join(genes) or '—'}** in **{cell}**, "
                "then download and run the notebook."
            )
            if plan.get("cell_reason"):
                st.caption(f"Why this cell context: {plan['cell_reason']}")
            if st.button("Prefill the New perturbation tab", key="sh_fill1"):
                st.session_state.genes_in = ", ".join(genes)
                st.session_state.preset_name = cell
                st.success("Filled. Open the New perturbation tab above.")

        with st.container(border=True):
            st.markdown("**2 · Generate & rank hypotheses**  —  Run Benchmate tab")
            st.markdown(
                f"Run **{iters} iterations** on your question — the agents "
                "propose hypotheses and rank them by Elo."
            )
            if st.button("Prefill the Run Benchmate tab", key="sh_fill2"):
                st.session_state.goal = question
                st.session_state.run_iterations = iters
                st.success("Filled. Open the Run Benchmate tab above.")

        with st.container(border=True):
            st.markdown("**3 · Cross-check the winners**  —  Cross-check tab")
            if xchecks:
                for x in xchecks:
                    st.markdown(
                        f"- **{str(x.get('target', '?')).upper()}** → "
                        f"*{x.get('model', '?')}*  ({x.get('reason', '')})"
                    )
            else:
                st.markdown("Score your top genes/proteins against the panel.")
            st.caption("Low agreement with the Elo ranking = a flag before the bench.")
            st.session_state.sh_crosscheck = xchecks

        if st.button("Prefill steps 1 & 2", key="sh_fill_all"):
            st.session_state.genes_in = ", ".join(genes)
            st.session_state.preset_name = cell
            st.session_state.goal = question
            st.session_state.run_iterations = iters
            st.success("Both tabs filled. Work through them top to bottom.")

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
            key="genes_in",
        )
    with col2:
        preset_name = st.selectbox(
            "Cell context",
            _PRESET_KEYS,
            key="preset_name",
            help="The cells Geneformer will perturb your genes in. "
                 "Pick where your genes' biology should be readable.",
        )
    if st.session_state.get("sh_plan"):
        st.caption("Tip: the Start here tab can fill these in from a question.")
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
        key="goal",
    )
    iterations = st.slider("Max iterations", 4, 16, key="run_iterations")
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
    st.markdown("### Live benchmarks (real API calls)")

    from benchmark import results as bench_results
    IS_HOSTED = bool(os.environ.get("BENCHMATE_HOSTED"))

    def _show_saved(name: str) -> None:
        """Render the latest saved run for a benchmark (params + metric tiles).
        On the hosted demo this is the only thing shown for a section."""
        run = bench_results.latest(name)
        if not run:
            if IS_HOSTED:
                st.caption("No saved run yet — run this locally to populate the "
                           "hosted demo.")
            return
        p = run.get("params", {})
        bits = ", ".join(f"{k}={v}" for k, v in p.items())
        st.caption(f"📊 Saved local run{(' · ' + bits) if bits else ''} · "
                   f"captured {run.get('captured', '?')}")
        metrics = run.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k, v)
        if run.get("note"):
            st.caption(run["note"])

    if IS_HOSTED:
        st.info(
            "**Hosted demo.** Each section shows results captured on a local "
            "run (saved in the repo). Want to run them live yourself? Enter "
            "**your own** Anthropic key in the sidebar — you pay only for your "
            "own calls — and the key-only benchmarks below unlock. The ontology "
            "comparison additionally needs a local OntoMCP server (see its "
            "section), so it can't run on the hosted site."
        )
    else:
        st.caption(
            "These run on the ERAD gold set in `benchmark/gold_set.py` and need "
            "an Anthropic key (set in the sidebar). Ranking calls use Haiku, "
            "so the spend is small — the estimates below are upper bounds. "
            "Each run is saved to `benchmark/results/`, which is what the "
            "hosted demo displays."
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning(
            "Add an Anthropic key in the sidebar to run any of these live. "
            "Without one, the saved results above each section are still shown."
        )

    # ---- 1. Judge accuracy ------------------------------------------------
    with st.container(border=True):
        st.markdown(
            "**1. Judge accuracy** — is the LLM judge any good? "
            "Measures accuracy on clear cross-tier pairs, position-bias rate "
            "(how often the verdict flips when you swap A/B), self-consistency, "
            "and transitivity violations."
        )
        _show_saved("judge_eval")
        je_pairs = st.slider("Cross-tier pairs to test", 4, 12, 8,
                             key="je_pairs",
                             help="More pairs = tighter estimates but more spend.")
        je_cost = 0.001 * (je_pairs * 4 + 30)        # ~4 calls/pair + transitivity sweep
        st.caption(f"Estimated Anthropic spend: ~${je_cost:.2f}. "
                   f"Expected runtime: ~{max(1, je_pairs // 2)}–"
                   f"{je_pairs} minutes.")
        if st.button("Run judge-eval", type="primary",
                     disabled=not os.environ.get("ANTHROPIC_API_KEY"),
                     key="je_btn"):
            from benchmark.judge_eval import evaluate_judge
            with st.spinner("Judging gold-set pairs in both orders…"):
                r = evaluate_judge(max_pairs=int(je_pairs), role="ranking")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("accuracy", f"{r.accuracy:.0%}",
                      help="Want > 80% on clear pairs.")
            c2.metric("position bias", f"{r.position_bias_rate:.0%}",
                      help="Want < 15%. Verdicts that flip with A/B order.")
            c3.metric("self-consistency", f"{r.self_consistency:.0%}",
                      help="Want > 85%. Same call twice agrees.")
            tv = (f"{r.transitivity_violations}/{r.transitivity_triples}"
                  if r.transitivity_triples else "n/a")
            c4.metric("transitivity", tv,
                      help="Want ~0 cycles (A>B>C>A).")
            bench_results.save_run("judge_eval", {
                "label": "Judge accuracy",
                "params": {"pairs": int(je_pairs)},
                "metrics": {"accuracy": f"{r.accuracy:.0%}",
                            "position bias": f"{r.position_bias_rate:.0%}",
                            "self-consistency": f"{r.self_consistency:.0%}",
                            "transitivity": tv},
            })
            if r.accuracy < 0.8 or r.position_bias_rate > 0.15:
                st.info(
                    "The judge is the bottleneck. The fair judge (default) "
                    "is already neutralising bias; a stronger ranking model "
                    "(Sonnet via `BENCHMATE_MODEL_RANKING`) is the next "
                    "lever."
                )

    # ---- 2. Validate vs gold ---------------------------------------------
    with st.container(border=True):
        st.markdown(
            "**2. Validate vs gold** — rank the ERAD gold set end-to-end "
            "and score the leaderboard against the known tier order. Use "
            "as a regression test after any prompt or model change."
        )
        _show_saved("validate")
        col_a, col_b = st.columns(2)
        with col_a:
            va_cycles = st.slider("Cycles", 2, 10, 6, key="va_cycles")
        with col_b:
            va_npc = st.slider("Matches per cycle", 4, 12, 8, key="va_npc")
        va_calls = va_cycles * va_npc * 2            # fair judge = 2 calls/match
        va_cost = 0.001 * va_calls
        st.caption(f"~{va_calls} judge calls. Estimated spend: ~${va_cost:.2f}. "
                   f"Runtime: a few minutes.")
        if st.button("Run validate", type="primary",
                     disabled=not os.environ.get("ANTHROPIC_API_KEY"),
                     key="va_btn"):
            from benchmark.run_benchmark import (run_tournament, _fair_judge_fn,
                                                  CRITERIA)
            from benchmark.gold_set import GOLD, gold_hypotheses
            from benchmark.metrics import spearman, topk_jaccard
            hyps = gold_hypotheses()
            with st.spinner(f"Running {va_cycles}×{va_npc} matches with the "
                             "fair judge…"):
                run_tournament(hyps, _fair_judge_fn(),
                                cycles=int(va_cycles),
                                n_per_cycle=int(va_npc))
            # Score against gold
            gold_rank_of = {h.id: i for i, h in enumerate(hyps)}
            gold_scores = [-gold_rank_of[h.id] for h in hyps]
            elo_scores = [h.elo for h in hyps]
            rho = spearman(gold_scores, elo_scores)
            final = sorted(hyps, key=lambda h: -h.elo)
            gold_order = [h.id for h in sorted(hyps,
                                               key=lambda h: gold_rank_of[h.id])]
            final_order = [h.id for h in final]
            top1 = final_order[0] == gold_order[0]
            top3 = topk_jaccard(final_order, gold_order, 3)

            c1, c2, c3 = st.columns(3)
            c1.metric("spearman vs gold", f"{rho:+.2f}",
                      help="1.0 = matches gold tier order exactly.")
            c2.metric("top-1 correct", "yes" if top1 else "no",
                      help="Did a tier-A hypothesis finish #1?")
            c3.metric("top-3 overlap", f"{top3:.0%}")
            bench_results.save_run("validate", {
                "label": "Validate vs gold",
                "params": {"cycles": int(va_cycles), "matches/cycle": int(va_npc)},
                "metrics": {"spearman vs gold": f"{rho:+.2f}",
                            "top-1 correct": "yes" if top1 else "no",
                            "top-3 overlap": f"{top3:.0%}"},
            })

            import pandas as pd
            rows = []
            for rank, h in enumerate(final, 1):
                tier = GOLD[next(i for i, x in enumerate(hyps)
                                 if x.id == h.id)]["tier"]
                rows.append({
                    "rank": rank,
                    "tier": tier,
                    "Elo": round(h.elo, 0),
                    "statement": h.statement,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

    # ---- 3. Compare fair vs naive ----------------------------------------
    with st.container(border=True):
        st.markdown(
            "**3. Compare fair vs naive judge** — same gold set, ranked "
            "under both judges. Δ Spearman tells you whether the "
            "order-swap is actually buying accuracy."
        )
        _show_saved("compare_fair_naive")
        col_a, col_b = st.columns(2)
        with col_a:
            cm_cycles = st.slider("Cycles", 2, 10, 6, key="cm_cycles")
        with col_b:
            cm_npc = st.slider("Matches per cycle", 4, 12, 8, key="cm_npc")
        cm_calls = cm_cycles * cm_npc * 3            # naive 1× + fair 2× per match
        cm_cost = 0.001 * cm_calls
        st.caption(f"~{cm_calls} judge calls (naive once + fair twice per "
                   f"match). Estimated spend: ~${cm_cost:.2f}.")
        if st.button("Run compare", type="primary",
                     disabled=not os.environ.get("ANTHROPIC_API_KEY"),
                     key="cm_btn"):
            from benchmark.run_benchmark import (run_tournament,
                                                  _fair_judge_fn,
                                                  _naive_judge_fn)
            from benchmark.gold_set import gold_hypotheses
            from benchmark.metrics import spearman

            def score(hyps):
                gold_rank_of = {h.id: i for i, h in enumerate(hyps)}
                return spearman([-gold_rank_of[h.id] for h in hyps],
                                [h.elo for h in hyps])

            with st.spinner("Running naive judge…"):
                h1 = gold_hypotheses()
                run_tournament(h1, _naive_judge_fn(),
                                cycles=int(cm_cycles),
                                n_per_cycle=int(cm_npc))
                rho_naive = score(h1)
            with st.spinner("Running fair judge…"):
                h2 = gold_hypotheses()
                run_tournament(h2, _fair_judge_fn(),
                                cycles=int(cm_cycles),
                                n_per_cycle=int(cm_npc))
                rho_fair = score(h2)

            c1, c2, c3 = st.columns(3)
            c1.metric("naive judge — spearman", f"{rho_naive:+.2f}")
            c2.metric("fair judge — spearman", f"{rho_fair:+.2f}")
            c3.metric("Δ (fair − naive)", f"{rho_fair - rho_naive:+.2f}",
                      delta=f"{rho_fair - rho_naive:+.2f}")
            bench_results.save_run("compare_fair_naive", {
                "label": "Fair vs naive judge",
                "params": {"cycles": int(cm_cycles), "matches/cycle": int(cm_npc)},
                "metrics": {"naive — spearman": f"{rho_naive:+.2f}",
                            "fair — spearman": f"{rho_fair:+.2f}",
                            "Δ (fair − naive)": f"{rho_fair - rho_naive:+.2f}"},
            })
            if rho_fair > rho_naive:
                st.success("Fair judge ranks closer to the gold tier order.")
            elif rho_fair < rho_naive:
                st.warning(
                    "Fair judge underperformed in this run. Re-run a few "
                    "times — a single comparison is one noisy sample. "
                    "Trust the median of 3+ runs."
                )

    # ---- 4. Ontology grounding (structured-knowledge layer) --------------
    with st.container(border=True):
        st.markdown(
            "**4. Ontology grounding** — the fair judge ranks the same gold "
            "set with vs without a canonical 'known-biology' block "
            "(genes / diseases / pathways resolved via "
            "[OntoMCP](https://github.com/jeanlouishoneine-tech/OntoMCP)) "
            "injected into the prompt. Δ Spearman tells you whether grounding "
            "in structured knowledge actually beats reading the text alone."
        )
        _show_saved("compare_ontology")

        from co_scientist.ontology import ontomcp_available, ONTOMCP_API_URL
        onto_up = ontomcp_available()
        if onto_up:
            st.caption(f"✓ OntoMCP reachable at {ONTOMCP_API_URL}")
        else:
            st.info(
                "This comparison needs a local **OntoMCP** server, which isn't "
                "part of the hosted demo. To run it yourself: clone and start "
                "OntoMCP in a separate terminal, add your Anthropic key in the "
                "sidebar, then reload.\n\n"
                "```\ngit clone https://github.com/jeanlouishoneine-tech/"
                "OntoMCP.git && cd OntoMCP\nuv sync && uv run ontomcp-api\n```\n"
                f"(Set `ONTOMCP_API_URL` if it isn't on {ONTOMCP_API_URL}.)"
            )
        col_a, col_b = st.columns(2)
        with col_a:
            on_cycles = st.slider("Cycles", 2, 10, 6, key="on_cycles")
        with col_b:
            on_npc = st.slider("Matches per cycle", 4, 12, 8, key="on_npc")
        on_calls = on_cycles * on_npc * 4        # fair judge twice, both arms
        st.caption(f"~{on_calls} judge calls (fair judge twice per match, "
                   f"both arms). Estimated spend: ~${0.001 * on_calls:.2f}. "
                   "OntoMCP lookups are local and free.")
        if st.button("Run ontology compare", type="primary",
                     disabled=not (os.environ.get("ANTHROPIC_API_KEY")
                                   and onto_up),
                     key="on_btn"):
            from benchmark.run_benchmark import run_tournament, _fair_judge_fn
            from benchmark.gold_set import gold_hypotheses
            from benchmark.metrics import spearman

            def score(hyps):
                gold_rank_of = {h.id: i for i, h in enumerate(hyps)}
                return spearman([-gold_rank_of[h.id] for h in hyps],
                                [h.elo for h in hyps])

            with st.spinner("Running fair judge, grounding OFF…"):
                h1 = gold_hypotheses()
                run_tournament(h1, _fair_judge_fn(ontology=False),
                                cycles=int(on_cycles), n_per_cycle=int(on_npc))
                rho_base = score(h1)
            with st.spinner("Running fair judge, grounding ON…"):
                h2 = gold_hypotheses()
                run_tournament(h2, _fair_judge_fn(ontology=True),
                                cycles=int(on_cycles), n_per_cycle=int(on_npc))
                rho_onto = score(h2)

            c1, c2, c3 = st.columns(3)
            c1.metric("grounding OFF — spearman", f"{rho_base:+.2f}")
            c2.metric("grounding ON — spearman", f"{rho_onto:+.2f}")
            c3.metric("Δ (ON − OFF)", f"{rho_onto - rho_base:+.2f}",
                      delta=f"{rho_onto - rho_base:+.2f}")
            bench_results.save_run("compare_ontology", {
                "label": "Ontology grounding (fair judge, OFF vs ON)",
                "params": {"cycles": int(on_cycles),
                           "matches/cycle": int(on_npc)},
                "metrics": {"grounding OFF — spearman": f"{rho_base:+.2f}",
                            "grounding ON — spearman": f"{rho_onto:+.2f}",
                            "Δ (ON − OFF)": f"{rho_onto - rho_base:+.2f}"},
            })
            if rho_onto > rho_base:
                st.success("Ontology grounding ranks closer to the gold "
                           "tier order.")
            elif rho_onto < rho_base:
                st.warning(
                    "Grounding underperformed in this run. One comparison "
                    "is one noisy sample — trust the median of 3+ runs."
                )
            else:
                st.info("No difference this run. Re-run a few times before "
                        "concluding.")

    # ---- 5. Ontology discrimination (traps vs novelty) -------------------
    with st.container(border=True):
        st.markdown(
            "**5. Ontology discrimination** — the honest version of #4. Ranks an "
            "adversarial set with three kinds of hypothesis (solid, fluent-but-"
            "**false** traps, and **novel**-but-true) and asks: does grounding "
            "sink the traps **without** punishing the novel ideas? "
            "*Trap demotion* should be positive; *novelty penalty* should be ~0."
        )
        _show_saved("discrimination")
        if not IS_HOSTED:
            d_cols = st.columns(2)
            with d_cols[0]:
                ad_cycles = st.slider("Cycles", 2, 10, 6, key="ad_cycles")
            with d_cols[1]:
                ad_npc = st.slider("Matches per cycle", 4, 12, 8, key="ad_npc")
            st.caption(f"~{ad_cycles * ad_npc * 4} judge calls (fair judge twice, "
                       f"both arms). Needs OntoMCP running.")
            if st.button("Run discrimination test", type="primary",
                         disabled=not (os.environ.get("ANTHROPIC_API_KEY")
                                       and onto_up),
                         key="ad_btn"):
                from benchmark.run_benchmark import run_tournament, _fair_judge_fn
                from benchmark.gold_set_adversarial import adversarial_hypotheses
                from benchmark.run_adversarial import summarize
                with st.spinner("Ranking adversarial set, grounding OFF…"):
                    off = adversarial_hypotheses()
                    run_tournament(off, _fair_judge_fn(ontology=False),
                                   int(ad_cycles), int(ad_npc))
                with st.spinner("Ranking adversarial set, grounding ON…"):
                    on = adversarial_hypotheses()
                    run_tournament(on, _fair_judge_fn(ontology=True),
                                   int(ad_cycles), int(ad_npc))
                res = summarize(off, on)
                c1, c2, c3 = st.columns(3)
                c1.metric("trap demotion", f"{res['trap_demotion']:+.1f}",
                          help="Want > 0 — grounding sinks the false traps.")
                c2.metric("novelty penalty", f"{res['novelty_penalty']:+.1f}",
                          help="Want ~0 — grounding leaves novel ideas alone.")
                c3.metric("spearman ON", f"{res['spearman_on']:+.2f}",
                          delta=f"{res['spearman_on'] - res['spearman_off']:+.2f}")
                bench_results.save_run("discrimination", {
                    "label": "Ontology discrimination",
                    "params": {"cycles": int(ad_cycles), "matches/cycle": int(ad_npc)},
                    "metrics": {"trap demotion": f"{res['trap_demotion']:+.1f}",
                                "novelty penalty": f"{res['novelty_penalty']:+.1f}",
                                "spearman ON": f"{res['spearman_on']:+.2f}"},
                })
                if res["trap_demotion"] > 0 and res["novelty_penalty"] <= 0.75:
                    st.success("Grounding caught the traps without punishing novelty.")
                elif res["trap_demotion"] <= 0:
                    st.warning("Grounding didn't demote the traps — coverage / "
                               "placement issue. (One run is one noisy sample.)")
                else:
                    st.warning("Traps sank but novel ideas dropped too — the "
                               "consensus-filter risk. Try grounding in review only.")

    # ---- 6. Alias / duplicate detection ----------------------------------
    with st.container(border=True):
        st.markdown(
            "**6. Alias / duplicate detection** — where the ontology beats text "
            "similarity. Two hypotheses can be the *same idea* worded differently "
            "(\"multiple myeloma\" vs \"plasma cell myeloma\"). The metric is "
            "**separation**: how much higher a method scores true duplicates than "
            "unrelated pairs. No API key needed — just embeddings + OntoMCP lookups."
        )
        _show_saved("alias_dedup")
        if st.button("Run alias-dedup", key="al_btn"):
            from benchmark.alias_dedup import (summarize, _verdict,
                                               _build_rows_live, _build_rows_demo)
            from co_scientist.ontology import ontomcp_available
            live = ontomcp_available() and not IS_HOSTED
            rows = _build_rows_live() if live else _build_rows_demo()
            st.caption("Source: live OntoMCP" if live
                       else "synthetic demo data (OntoMCP not reachable / hosted)")
            res = summarize(rows)
            c1, c2 = st.columns(2)
            c1.metric("ontology separation", f"{res['ontology_separation']:+.2f}",
                      help="same − different. Higher = better at spotting duplicates.")
            c2.metric("embedding separation", f"{res['embedding_separation']:+.2f}")
            import pandas as pd
            df = pd.DataFrame(
                [{"pair": r["kind"], "label": r["label"],
                  "ontology": round(r["onto"], 2), "embedding": round(r["emb"], 2),
                  "note": ("⚠ not unified" if r.get("onto_na") else "")}
                 for r in rows])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(_verdict(res))
            bench_results.save_run("alias_dedup", {
                "label": "Alias / duplicate detection",
                "params": {"pairs": len(rows)},
                "metrics": {"ontology separation": f"{res['ontology_separation']:+.2f}",
                            "embedding separation": f"{res['embedding_separation']:+.2f}"},
            })

    st.divider()
    st.caption(
        "Full plan and recommended sequence of work: "
        "[benchmark/BENCHMARKING_PLAN.md]"
        "(https://github.com/nataliegits/Benchmate/blob/main/benchmark/"
        "BENCHMARKING_PLAN.md). Gold set lives in `benchmark/gold_set.py` "
        "(currently ERAD / bortezomib-resistant multiple myeloma). "
        "The same benchmarks also run as CLI commands "
        "(`python -m benchmark.run_benchmark <subcommand>`)."
    )

# ── Tab 5 — Cross-check with other models ─────────────────────
with tab5:
    from benchmark import results as bench_results
    from benchmark.elo_vs_variant_score import correlate, _load, _demo
    IS_HOSTED = bool(os.environ.get("BENCHMATE_HOSTED"))

    def _show_saved(name: str) -> None:
        run = bench_results.latest(name)
        if not run:
            if IS_HOSTED:
                st.caption("No saved run yet — run this locally to populate the demo.")
            return
        p = run.get("params", {})
        bits = ", ".join(f"{k}={v}" for k, v in p.items())
        st.caption(f"📊 Saved run{(' · ' + bits) if bits else ''} · "
                   f"captured {run.get('captured', '?')}")
        metrics = run.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k, v)

    def _show_correlation(elo, score, labels, src, save_key, score_col):
        import pandas as pd
        res = correlate(elo, score)
        st.caption(f"Source: {src}")
        if res["spearman"] is None:
            st.info(res["note"]); return
        st.metric("Spearman(Elo, predictor)", f"{res['spearman']:+.2f}")
        st.caption(res["verdict"])
        df = pd.DataFrame(sorted(zip(labels, elo, score), key=lambda t: -t[1]),
                          columns=["hypothesis", "Elo", score_col])
        st.dataframe(df, use_container_width=True, hide_index=True)
        bench_results.save_run(save_key, {
            "label": save_key, "params": {"n": res["n"]},
            "metrics": {"Spearman(Elo, predictor)": f"{res['spearman']:+.2f}"}})

    st.markdown("### Cross-check with other models")
    st.markdown(
        "The Elo leaderboard is the LLM judge's opinion. Before trusting it to pick "
        "a wet-lab candidate, cross-check it against **independent quantitative "
        "models** — each scoring a hypothesis from a completely different angle. "
        "Low correlation = a flag. This is the *panel of judges*."
    )

    _sh_xc = st.session_state.get("sh_crosscheck")
    if _sh_xc:
        lines = "\n".join(
            f"- **{str(x.get('target', '?')).upper()}** → "
            f"*{x.get('model', '?')}* ({x.get('reason', '')})"
            for x in _sh_xc
        )
        st.info("From your Start here plan, check these:\n\n" + lines)

    # ----- AlphaGenome: regulatory / expression -----
    with st.container(border=True):
        st.markdown(
            "**AlphaGenome — regulatory effect** *(does a variant change expression?)*. "
            "Drop `variant_scores.json` (merged Elo + score) or a raw "
            "`alphagenome_scores.json` from Colab. See `benchmark/ALPHAGENOME_PLAN.md`."
        )
        _show_saved("elo_vs_predictor")
        with st.expander("How to produce real scores (AlphaGenome setup)"):
            st.markdown(
                "1. Free key at [alphagenomedocs.com](https://www.alphagenomedocs.com/) "
                "→ *Get started* (sign in with a personal @gmail).\n"
                "2. Score in `benchmark/alphagenome_scoring_colab.ipynb` on "
                "[Colab](https://colab.research.google.com) → downloads "
                "`alphagenome_scores.json`.\n"
                "3. Put it in `benchmark/`, run `python -m benchmark.build_variant_scores`.\n"
                "⚠️ Variant coordinates: run `python -m benchmark.fetch_eqtls` for real GTEx eQTLs."
            )
        up = st.file_uploader("Drop alphagenome_scores.json or variant_scores.json",
                              type=["json"], key="ag_up")
        if st.button("Show AlphaGenome correlation", key="ag_btn"):
            import json
            elo = score = labels = None; src = ""
            if up is not None:
                data = json.load(up)
                if isinstance(data, list):
                    elo = [float(r["elo"]) for r in data]
                    score = [float(r["score"]) for r in data]
                    labels = [str(r.get("label", i)) for i, r in enumerate(data)]
                    src = "uploaded variant_scores.json"
                elif isinstance(data, dict):
                    from benchmark.gold_set_variants import GOLD_VARIANTS
                    if os.environ.get("ANTHROPIC_API_KEY") and not IS_HOSTED:
                        from benchmark.build_variant_scores import _elo_by_label
                        with st.spinner("Ranking variant hypotheses for Elo…"):
                            em = _elo_by_label(6, 8)
                        trip = [(g["label"], em[g["label"]], float(data[g["label"]]))
                                for g in GOLD_VARIANTS
                                if g["label"] in data and g["label"] in em]
                        labels = [t[0] for t in trip]; elo = [t[1] for t in trip]
                        score = [t[2] for t in trip]
                        src = "uploaded AlphaGenome scores + freshly-ranked Elo"
                    else:
                        st.warning("Raw scores need an Elo column — set your Anthropic "
                                   "key, run build_variant_scores locally, or upload "
                                   "the merged variant_scores.json.")
            else:
                p = "benchmark/variant_scores.json"
                if os.path.exists(p):
                    elo, score, labels = _load(p); src = f"`{p}`"
                else:
                    elo, score, labels = _demo(); src = "synthetic demo data"
            if elo is not None:
                _show_correlation(elo, score, labels, src,
                                  "elo_vs_predictor", "AlphaGenome score")

    # ----- Boltz: structure / binding -----
    with st.container(border=True):
        st.markdown(
            "**Boltz — structure & binding** *(does the drug actually bind the "
            "target?)*. The complement to AlphaGenome: it scores binding-style "
            "hypotheses. Drop a `boltz_scores.json` (merged Elo + score)."
        )
        _show_saved("elo_vs_boltz")
        with st.expander("How to produce Boltz scores"):
            st.markdown(
                "1. Sign up + redeem **$100 credits (code BOLTZLAUNCH)** at "
                "[api.boltz.bio](https://api.boltz.bio/console/signup), create an API "
                "key, `export BOLTZ_API_KEY=...`.\n"
                "2. For each binding hypothesis, score the protein+ligand with "
                "`co_scientist/boltz_scorer.py` (`BoltzTarget` → `score_binding`).\n"
                "3. Save `benchmark/boltz_scores.json` as "
                "`[{label, elo, score}, ...]`.\n"
                "⚠️ The Boltz API is new — verify the endpoints in boltz_scorer.py "
                "against the console's API reference."
            )
        upb = st.file_uploader("Drop boltz_scores.json", type=["json"], key="bz_up")
        if st.button("Show Boltz correlation", key="bz_btn"):
            import json
            elo = score = labels = None; src = ""
            if upb is not None:
                data = json.load(upb)
                if isinstance(data, list):
                    elo = [float(r["elo"]) for r in data]
                    score = [float(r["score"]) for r in data]
                    labels = [str(r.get("label", i)) for i, r in enumerate(data)]
                    src = "uploaded boltz_scores.json"
                else:
                    st.warning("Upload a merged list of {label, elo, score}.")
            else:
                p = "benchmark/boltz_scores.json"
                if os.path.exists(p):
                    elo, score, labels = _load(p); src = f"`{p}`"
                else:
                    st.info("No boltz_scores.json yet — see the setup steps above.")
            if elo is not None:
                _show_correlation(elo, score, labels, src,
                                  "elo_vs_boltz", "Boltz score")

    # ----- generic gene/variant judges (Open Targets, DepMap, AlphaMissense) -----
    def _simple_panel(title, desc, default_path, save_key, up_key, btn_key,
                      score_col, setup_md=None):
        with st.container(border=True):
            st.markdown(f"**{title}** — {desc}")
            _show_saved(save_key)
            if setup_md:
                with st.expander("How to produce these scores"):
                    st.markdown(setup_md)
            up = st.file_uploader(f"Drop {default_path.split('/')[-1]}",
                                  type=["json"], key=up_key)
            if st.button(f"Show {title} correlation", key=btn_key):
                import json
                elo = score = labels = None; src = ""
                if up is not None:
                    data = json.load(up)
                    if isinstance(data, list):
                        elo = [float(r["elo"]) for r in data]
                        score = [float(r["score"]) for r in data]
                        labels = [str(r.get("label", i)) for i, r in enumerate(data)]
                        src = f"uploaded {default_path.split('/')[-1]}"
                    else:
                        st.warning("Upload a merged list of {label, elo, score}.")
                elif os.path.exists(default_path):
                    elo, score, labels = _load(default_path); src = f"`{default_path}`"
                else:
                    st.info("No scores file yet — see the setup steps above.")
                if elo is not None:
                    _show_correlation(elo, score, labels, src, save_key, score_col)

    _simple_panel(
        "Open Targets — disease association",
        "*is this gene genuinely linked to the disease?* (genetics + literature + drugs). "
        "Free, no key.",
        "benchmark/opentargets_scores.json", "elo_vs_opentargets",
        "ot_up", "ot_btn", "association",
        "Run `python -m benchmark.build_target_scores` (needs your Anthropic key; "
        "Open Targets is a free API). It writes `opentargets_scores.json`.")

    _simple_panel(
        "DepMap — gene dependency",
        "*is this gene actually essential in the disease's cancer cell lines?*",
        "benchmark/depmap_scores.json", "elo_vs_depmap",
        "dm_up", "dm_btn", "dependency",
        "Download `CRISPRGeneEffect.csv` from [depmap.org]"
        "(https://depmap.org/portal/data_page/) into `data/depmap/`, then run "
        "`python -m benchmark.build_target_scores`. **Optional:** also drop "
        "`Model.csv` there and the score is restricted to multiple-myeloma cell "
        "lines instead of pan-cancer.")

    _simple_panel(
        "AlphaMissense — variant pathogenicity",
        "*is a coding variant likely pathogenic?* (free, via Ensembl VEP). "
        "Scores missense variants — supply real ones.",
        "benchmark/alphamissense_scores.json", "elo_vs_alphamissense",
        "am_up", "am_btn", "pathogenicity",
        "1. `python -m benchmark.fetch_clinvar` pulls **real** pathogenic + benign "
        "missense variants (GRCh38 coords) from ClinVar. 2. "
        "`python -m benchmark.build_missense_scores` ranks them by Elo, scores each "
        "with AlphaMissense (free Ensembl VEP), and writes "
        "`alphamissense_scores.json` — plus a pathogenic-vs-benign calibration check.")
