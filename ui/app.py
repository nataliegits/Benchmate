"""Streamlit UI for Benchmate's "add genes → Colab → Benchmate" loop.

Run with:
    cd ~/Desktop/Benchmate
    source .venv/bin/activate
    streamlit run ui/app.py
"""
from __future__ import annotations

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

st.set_page_config(page_title="Benchmate", page_icon="🧪", layout="wide")
st.title("🧪 Benchmate")
st.caption("AI co-scientist for biomedical hypothesis generation, grounded in your own perturbation data.")

# ────────────────────────────────────────────────────────────
# Sidebar: API key + cache status
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🔑 Anthropic API key")
    # Use session state so the key persists across tab switches
    if "anthropic_key" not in st.session_state:
        st.session_state.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    st.session_state.anthropic_key = st.text_input(
        "Your Anthropic key",
        value=st.session_state.anthropic_key,
        type="password",
        help="Get one at console.anthropic.com. Stays in your browser session "
             "only — never sent to the Benchmate maintainer.",
    )
    if st.session_state.anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key
        st.caption("✅ Key set for this session.")
    else:
        st.caption("⚠️ Required for tab 3 (running the agent loop).")

    st.divider()
    st.subheader("📁 Cached perturbations")
    cached = available_geneformer_genes()
    if cached:
        for g in cached:
            st.markdown(f"• `{g}`")
    else:
        st.info("No cached perturbations yet.")

    st.divider()
    st.subheader("⬆ Upload CSVs")
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
                st.warning(f"Skipping {name} (expected `*_stats.csv`)")
                continue
            dst = CACHE_DIR / name
            dst.write_bytes(f.read())
            st.success(f"✓ cached {name}")
        st.rerun()

    st.divider()
    with st.expander("🧮 Model routing (Pi)", expanded=False):
        st.caption("Pick which model handles each agent role. Sonnet for "
                   "reasoning-heavy roles; Haiku/Flash for throughput. Changes "
                   "apply to the next Run Benchmate.")
        # Common multi-provider options via litellm
        MODEL_OPTIONS = [
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-opus-4-7",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash",
        ]
        ROLE_PRICE = {  # rough $/1M input tokens for cost hints
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
                help=f"~${ROLE_PRICE.get(current, '?')}/1M tok currently",
            )
            if choice != current:
                os.environ[f"BENCHMATE_MODEL_{role.upper()}"] = choice
        st.caption("_Tip:_ keep Generation/Reflection/Evolution on a strong "
                   "model (they read Geneformer evidence); the others can be "
                   "Haiku or Flash without losing quality.")

# ────────────────────────────────────────────────────────────
# Tab 1: Generate a Colab notebook for new genes
# ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "1. New perturbation",
    "2. Inspect cache",
    "3. Run Benchmate",
    "4. Hermes preview",
])

with tab1:
    st.header("Add genes to the perturbation cache")
    st.write("Type the gene symbols you want to perturb, pick the cell "
             "context to perturb them in, and we'll generate a Colab notebook "
             "with both pre-filled.")

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
            help="Pick the cells Geneformer will perturb your genes in. "
                 "Choose based on where your genes' biology should be readable.",
        )
    st.caption(f"_{CELL_TYPE_PRESETS[preset_name]['rationale']}_")

    if st.button("Generate Colab notebook", type="primary"):
        if not genes_in.strip():
            st.error("Enter at least one gene symbol.")
        else:
            symbols = [g.strip().upper() for g in genes_in.split(",") if g.strip()]
            with st.spinner("Resolving Ensembl IDs…"):
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

            with st.spinner("Generating notebook…"):
                nb_path, _ = generate_notebook(resolved.keys(), preset_name=preset_name)
            st.write(f"📓 Notebook saved: `{nb_path.relative_to(REPO_ROOT)}`")

            # Try Gist push (one-click Colab); fall back to download button
            if gh_available():
                with st.spinner("Pushing to Gist for one-click Colab…"):
                    try:
                        urls = handoff(nb_path, description=f"Benchmate: {', '.join(resolved)}")
                    except Exception as e:
                        urls = None
                        st.warning(f"Gist push failed: {e}. Falling back to download.")
                if urls:
                    st.markdown(f"### [▶ Open in Colab]({urls['colab_url']})")
                    st.caption(f"Gist: {urls['gist_url']}")
            else:
                st.info(
                    "`gh` CLI not available — download the notebook below, "
                    "open colab.research.google.com → File → Upload notebook, "
                    "drop it in."
                )
                st.download_button(
                    "⬇ Download notebook",
                    nb_path.read_bytes(),
                    file_name=nb_path.name,
                    mime="application/x-ipynb+json",
                )

            st.divider()
            st.info(
                "In Colab: **Runtime → Change runtime type → T4 GPU**, then "
                "**Runtime → Run all**. When it finishes, the final cell "
                "auto-downloads each `*_stats.csv` to your browser. Drag those "
                "files into the **Upload CSVs** box in the sidebar — that's "
                "it, no Drive required."
            )

# ────────────────────────────────────────────────────────────
# Tab 2: Browse cached perturbations
# ────────────────────────────────────────────────────────────
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
            st.write(f"Top {r['n_results']} affected genes when **{gene}** was deleted:")
            import pandas as pd
            df = pd.DataFrame(r["affected_genes"])
            st.dataframe(df, use_container_width=True)

# ────────────────────────────────────────────────────────────
# Tab 3: Run the Benchmate loop
# ────────────────────────────────────────────────────────────
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
    cost_estimate = 0.30 * iterations / 4  # ~$0.30 per 4 iterations
    st.caption(f"Estimated Anthropic spend: **~${cost_estimate:.2f}** "
               f"({iterations} iterations).")

    if st.button("Run Benchmate", type="primary"):
        st.info("Running… this will stream output below. The Streamlit page "
                "stays responsive; you can browse other tabs.")
        log_box = st.empty()
        log_lines: list[str] = []
        proc = subprocess.Popen(
            [sys.executable, "run.py", goal, "--max-iterations", str(iterations)],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            log_lines.append(line.rstrip())
            log_box.code("\n".join(log_lines[-40:]))
        proc.wait()
        if proc.returncode == 0:
            st.success("Benchmate finished.")
            state_file = REPO_ROOT / "state.json"
            if state_file.exists():
                st.download_button(
                    "Download state.json",
                    state_file.read_text(),
                    file_name="state.json",
                )
        else:
            st.error(f"Benchmate exited with code {proc.returncode}.")


# ────────────────────────────────────────────────────────────
# Tab 4: Hermes preview — see what Hermes would receive
# ────────────────────────────────────────────────────────────
with tab4:
    st.header("Hermes preview")
    st.write("Hermes (https://hermes-agent.nousresearch.com/) is an autonomous "
             "agent that lives on a server and accepts messages from "
             "Slack/Discord/Telegram. When you wire it up to Benchmate, it "
             "calls the same JSON API previewed below.")
    st.caption("This tab just lets you _see_ what Hermes would receive without "
               "needing a VPS. Full deployment guide: `HERMES.md`.")

    import json
    from hermes.benchmate_runner import (
        list_cache as hermes_list_cache,
        gene_neighbors as hermes_neighbors,
        add_perturbation as hermes_add,
    )

    with st.container(border=True):
        st.markdown("**`list_cache()`** — what genes does Hermes 'see'?")
        st.caption("Slack equivalent: _\"hey hermes, what perturbations do we "
                   "have cached?\"_")
        if st.button("Run list_cache", key="hermes_list"):
            result = hermes_list_cache()
            st.code(json.dumps(result, indent=2), language="json")

    with st.container(border=True):
        st.markdown("**`gene_neighbors(symbol, top_n)`** — what would a "
                    "knockout perturb?")
        st.caption("Slack equivalent: _\"hermes, what does TXNDC15 knockout "
                   "shift downstream?\"_")
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
        st.markdown("**`add_perturbation(symbols, cell_context)`** — generate "
                    "a new Colab notebook")
        st.caption("Slack equivalent: _\"hermes, add FOXP3 to the perturbation "
                   "queue in plasma cells.\"_")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            new_genes = st.text_input("Gene symbols (comma-separated)",
                                      value="FOXP3", key="hermes_add_genes")
        with col_b:
            new_ctx = st.selectbox("Cell context",
                                   list(CELL_TYPE_PRESETS.keys()),
                                   index=2,  # plasma cells
                                   key="hermes_add_ctx")
        if st.button("Run add_perturbation", key="hermes_add_btn"):
            syms = [g.strip().upper() for g in new_genes.split(",") if g.strip()]
            result = hermes_add(syms, cell_context=new_ctx)
            st.code(json.dumps(result, indent=2, default=str), language="json")

    with st.container(border=True):
        st.markdown("**`run_benchmate(goal, max_iterations)`** — execute the "
                    "full agent loop")
        st.caption("Slack equivalent: _\"hermes, run benchmate on this goal: "
                   "...\"_")
        st.info("This one actually costs money (~$0.30–$3 depending on "
                "iterations). Use **Tab 3** to run it for real with live "
                "log streaming. The JSON output shape Hermes receives is:")
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
        "### Want this in Slack?\n"
        "All four commands above also work as CLI: "
        "`python -m hermes.benchmate_runner list-cache`. "
        "To make them chat-driven, deploy Hermes on a small VPS and "
        "register a skill that wraps the CLI. Full guide in "
        "[`HERMES.md`](https://github.com/nataliegits/Benchmate/blob/main/HERMES.md)."
    )
