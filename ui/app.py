"""Streamlit UI for Benchmate's "add genes → Colab → Benchmate" loop.

Run with:
    cd ~/Desktop/Benchmate
    source .venv/bin/activate
    streamlit run ui/app.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

# Make repo imports work whether this is run from repo root or ui/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from co_scientist.tools import available_geneformer_genes, geneformer_neighbors
from ui.notebook_gen import generate_notebook, resolve_to_ensembl, CELL_TYPE_PRESETS
from ui.colab_handoff import handoff


REPO_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Benchmate", page_icon="🧪", layout="wide")
st.title("🧪 Benchmate")
st.caption("AI co-scientist for biomedical hypothesis generation, grounded in your own perturbation data.")

# ────────────────────────────────────────────────────────────
# Sidebar: cache status
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Cached perturbations")
    cached = available_geneformer_genes()
    if cached:
        for g in cached:
            st.markdown(f"• `{g}`")
    else:
        st.info("No cached perturbations yet. Run the pipeline below to add some.")
    st.divider()
    st.caption("Drop `*_stats.csv` files into `data/geneformer/` "
               "(manually or via the watcher) to populate the cache.")

# ────────────────────────────────────────────────────────────
# Tab 1: Generate a Colab notebook for new genes
# ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "1. New perturbation",
    "2. Inspect cache",
    "3. Run Benchmate",
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

            with st.spinner("Pushing to Gist…"):
                try:
                    urls = handoff(nb_path, description=f"Benchmate: {', '.join(resolved)}")
                except Exception as e:
                    st.error(f"Gist push failed: {e}")
                    st.info("You can still upload the notebook to Colab manually.")
                    st.stop()
            st.markdown(f"### [▶ Open in Colab]({urls['colab_url']})")
            st.caption(f"Gist: {urls['gist_url']}")
            st.divider()
            st.info(
                "In Colab: **Runtime → Change runtime type → T4 GPU**, then "
                "**Runtime → Run all**. CSVs will land in your Google Drive "
                "under `MyDrive/benchmate_geneformer_cilia/perturbations/`. "
                "Run `python -m ui.watcher` locally to auto-copy them into the "
                "cache, or copy them manually into `data/geneformer/`."
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
