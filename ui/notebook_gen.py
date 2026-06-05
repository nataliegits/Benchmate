"""Generate a parameterised copy of notebook 02 with user-specified TARGETS
and a chosen cell-type preset.

The template's intro markdown, CELLxGENE pull cell, and TARGETS dict are all
rewritten based on user input. Everything else (install cells, perturbation,
stats, enrichment) is left untouched.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Iterable

import mygene

TEMPLATE = Path(__file__).resolve().parent.parent / "notebooks" / "02_geneformer_ciliated_cells.ipynb"
OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────
# Cell-type presets
# ────────────────────────────────────────────────────────────
# Each preset defines a CELLxGENE Census filter and the rationale that
# shows up in the notebook's intro markdown cell.
#
# To add a new preset: append an entry below. UI picks it up automatically.

CELL_TYPE_PRESETS: dict[str, dict] = {
    "Ciliated cells": {
        "cell_types": [
            "multi-ciliated epithelial cell",
            "ependymal cell",
            "kidney proximal tubule epithelial cell",
        ],
        "tissues": ["lung", "brain", "kidney"],
        "rationale": (
            "Multi-ciliated airway, ependymal, and kidney tubule cells. "
            "Use when your genes have ciliogenesis or ciliopathy biology "
            "(Meckel syndrome, polycystic kidney, primary ciliary dyskinesia)."
        ),
    },
    "Hepatocytes (ERAD-heavy)": {
        "cell_types": ["hepatocyte"],
        "tissues": ["liver"],
        "rationale": (
            "Hepatocytes from healthy human liver. Strong secretory load and "
            "constitutive ERAD; useful for ER protein-quality-control genes "
            "in a homogeneous, well-characterised context."
        ),
    },
    "Plasma cells (extreme ER stress)": {
        "cell_types": ["plasma cell"],
        "tissues": ["bone marrow", "blood"],
        "rationale": (
            "Antibody-secreting plasma cells live at the edge of the UPR. "
            "If your gene's loss-of-function produces an ERAD or proteostasis "
            "phenotype, the signal will be biggest here."
        ),
    },
    "Stem cells (pluripotent + adult)": {
        "cell_types": [
            "stem cell",
            "embryonic stem cell",
            "neural stem cell",
            "hematopoietic stem cell",
        ],
        "tissues": None,  # any tissue
        "rationale": (
            "Pluripotent and adult stem cells across tissues. Use for genes "
            "with developmental, self-renewal, or lineage-commitment biology."
        ),
    },
    "Neurons": {
        "cell_types": [
            "neuron",
            "GABAergic neuron",
            "glutamatergic neuron",
            "dopaminergic neuron",
        ],
        "tissues": ["brain", "spinal cord"],
        "rationale": (
            "CNS neurons across cortical, hippocampal, and brainstem regions. "
            "Use for proteostasis genes with neurodegeneration relevance "
            "(α-synuclein, tau, polyQ pathways)."
        ),
    },
    "Cancer cells (malignant)": {
        "cell_types": ["malignant cell"],
        "tissues": None,
        "rationale": (
            "Cancer cells across primary tumours in CELLxGENE. Use when your "
            "hypotheses relate to oncogenic context, dependency screens, or "
            "DepMap-style vulnerabilities."
        ),
    },
    "All cells (broad)": {
        "cell_types": None,  # no cell_type filter — let CELLxGENE return everything
        "tissues": None,
        "rationale": (
            "No cell-type filter; samples broadly across tissues. Maximum "
            "diversity but noisier signal. Use when you want Geneformer to "
            "integrate across the full atlas rather than focus."
        ),
    },
}


def _cell_filter_clause(preset: dict) -> str:
    """Build a CELLxGENE `obs_value_filter` clause from a preset config."""
    parts = []
    if preset["cell_types"]:
        ct_list = repr(preset["cell_types"])
        parts.append(f"cell_type in {ct_list}")
    if preset["tissues"]:
        t_list = repr(preset["tissues"])
        parts.append(f"tissue_general in {t_list}")
    parts.append("disease == 'normal'")
    parts.append("is_primary_data == True")
    return " and ".join(parts)


def resolve_to_ensembl(symbols: Iterable[str]) -> dict[str, str]:
    """Look up Ensembl gene IDs for each symbol via mygene."""
    mg = mygene.MyGeneInfo()
    out: dict[str, str] = {}
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        hits = mg.query(sym, fields="symbol,ensembl.gene", species="human").get("hits", [])
        for h in hits:
            if h.get("symbol") == sym:
                ens = h.get("ensembl", {})
                eid = (ens.get("gene") if isinstance(ens, dict)
                       else ens[0]["gene"] if ens else None)
                if eid:
                    out[sym] = eid
                    break
    return out


# ────────────────────────────────────────────────────────────
# Notebook surgery
# ────────────────────────────────────────────────────────────

def _build_intro_md(targets: dict[str, str], preset_name: str, preset: dict) -> list[str]:
    """Generate the intro markdown cell — gene-agnostic + preset-aware."""
    gene_lines = "\n".join(f"- **{sym}** ({eid})" for sym, eid in targets.items())
    return [
        "# Geneformer perturbation\n",
        "\n",
        "*Generated by Benchmate.*\n",
        "\n",
        f"## Targets\n",
        f"{gene_lines}\n",
        "\n",
        "Gene IDs were resolved via `mygene` before notebook generation, so "
        "the symbols and Ensembl IDs above are the canonical pairing.\n",
        "\n",
        f"## Cell context — {preset_name}\n",
        f"{preset['rationale']}\n",
        "\n",
        "## What this notebook does\n",
        "1. Pulls cells matching the chosen context from CELLxGENE Census.\n",
        "2. Tokenises them for Geneformer.\n",
        "3. Runs in-silico KO perturbation on each target.\n",
        "4. Aggregates per-gene shifts, computes pairwise and N-way "
        "intersections.\n",
        "5. Runs Enrichr (GO / Reactome / KEGG) on per-perturbation top hits "
        "and on intersections.\n",
        "\n",
        "**Caveat.** Geneformer's in-silico KO is a learned embedding shift, "
        "not a simulation of biology. Treat outputs as ranked hypotheses for "
        "wet-lab follow-up, not as conclusions.\n",
    ]


def _build_cellxgene_cell(preset: dict) -> list[str]:
    """Generate the CELLxGENE pull cell with the chosen filter."""
    filter_clause = _cell_filter_clause(preset)
    return [
        "import cellxgene_census\n",
        "import numpy as np\n",
        "\n",
        "CENSUS_VERSION = \"2024-07-01\"\n",
        "N_CELLS = 6000\n",
        "\n",
        f"OBS_FILTER = \"{filter_clause}\"\n",
        "print(\"CELLxGENE filter:\", OBS_FILTER)\n",
        "\n",
        "with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:\n",
        "    adata = cellxgene_census.get_anndata(\n",
        "        census=census,\n",
        "        organism=\"Homo sapiens\",\n",
        "        obs_value_filter=OBS_FILTER,\n",
        "        column_names={\n",
        "            \"obs\": [\"cell_type\", \"tissue\", \"disease\", \"assay\", \"donor_id\"],\n",
        "            \"var\": [\"feature_id\", \"feature_name\"],\n",
        "        },\n",
        "    )\n",
        "\n",
        "print(f\"Pulled {adata.n_obs} cells across {adata.obs['cell_type'].nunique()} cell types\")\n",
        "print(\"\\nCell type distribution:\")\n",
        "print(adata.obs[\"cell_type\"].value_counts().head(10))\n",
        "\n",
        "if adata.n_obs > N_CELLS:\n",
        "    idx = np.random.default_rng(0).choice(adata.n_obs, N_CELLS, replace=False)\n",
        "    adata = adata[idx].copy()\n",
        "    print(f\"\\nDown-sampled to {adata.n_obs} cells\")\n",
        "\n",
        "adata.var[\"ensembl_id\"] = adata.var[\"feature_id\"]\n",
        "adata.obs[\"n_counts\"] = adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else adata.X.sum(axis=1)\n",
        "\n",
        "RAW_PATH = f\"{OUT}/cells_raw.h5ad\"\n",
        "adata.write_h5ad(RAW_PATH)\n",
        "print(\"\\nsaved:\", RAW_PATH)\n",
    ]


def _build_download_cell(target_symbols: list[str]) -> list[str]:
    """Final cell that triggers a browser download of each *_stats.csv.

    Eliminates the Google Drive round-trip: the user gets the CSVs straight
    onto their Mac and uploads them into Benchmate via the Streamlit widget.
    """
    return [
        "# Auto-download perturbation CSVs to your machine.\n",
        "# Upload them via Benchmate's 'Upload CSVs' panel afterwards.\n",
        "from google.colab import files\n",
        "import os\n",
        "\n",
        f"DOWNLOAD_TARGETS = {target_symbols}\n",
        "for sym in DOWNLOAD_TARGETS:\n",
        "    csv_path = f\"{PERTURB_OUT}/{sym}/{sym}_stats.csv\"\n",
        "    if os.path.exists(csv_path):\n",
        "        print(f\"Downloading {sym}_stats.csv\")\n",
        "        files.download(csv_path)\n",
        "    else:\n",
        "        print(f\"NOT FOUND: {csv_path} (perturbation may have errored)\")\n",
        "print(\"\\nAll downloads triggered. Check your browser's Downloads folder.\")\n",
    ]


def generate_notebook(symbols: Iterable[str],
                      preset_name: str = "Ciliated cells",
                      out_dir: Path | None = None) -> tuple[Path, dict[str, str]]:
    """Write a parameterised notebook for the given gene symbols + cell preset.

    Returns (notebook_path, resolved_ensembl_map).
    Raises ValueError if no symbols could be resolved or preset is unknown.
    """
    if preset_name not in CELL_TYPE_PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. "
                         f"Available: {list(CELL_TYPE_PRESETS)}")
    preset = CELL_TYPE_PRESETS[preset_name]

    targets = resolve_to_ensembl(symbols)
    if not targets:
        raise ValueError(f"Could not resolve any of {list(symbols)} to Ensembl IDs.")

    nb = json.loads(TEMPLATE.read_text())

    intro_md = _build_intro_md(targets, preset_name, preset)
    cellxgene_src = _build_cellxgene_cell(preset)
    target_lines = ["TARGETS = {\n"]
    for sym, eid in targets.items():
        target_lines.append(f'    "{sym}":  "{eid}",\n')
    target_lines.append("}\n")

    # Replace, in order:
    #   - the first markdown cell (intro) -> intro_md
    #   - the cell containing `cellxgene_census.open_soma(` -> cellxgene_src
    #   - the cell containing `TARGETS = {` -> target_lines
    replaced = {"intro": False, "cellxgene": False, "targets": False}
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if not replaced["intro"] and cell.get("cell_type") == "markdown":
            cell["source"] = intro_md
            replaced["intro"] = True
            continue
        if not replaced["cellxgene"] and "cellxgene_census.open_soma" in src:
            cell["source"] = cellxgene_src
            replaced["cellxgene"] = True
            continue
        if not replaced["targets"] and re.search(r"TARGETS\s*=\s*\{", src):
            # Preserve any imports/preamble before the TARGETS literal
            pre_match = re.search(r"^(.*?)(TARGETS\s*=\s*\{)", src, re.DOTALL)
            preamble = pre_match.group(1) if pre_match else ""
            after_match = re.search(r"TARGETS\s*=\s*\{.*?\n\}(.*)$", src, re.DOTALL)
            tail = after_match.group(1) if after_match else ""
            cell["source"] = (
                ([preamble] if preamble else [])
                + target_lines
                + ([tail] if tail else [])
            )
            replaced["targets"] = True

    if not replaced["intro"]:
        raise RuntimeError("Template has no markdown cell to use as intro.")
    if not replaced["cellxgene"]:
        raise RuntimeError("Template lacks a CELLxGENE pull cell.")
    if not replaced["targets"]:
        raise RuntimeError("Template lacks a TARGETS = {...} cell.")

    # Append an auto-download cell so the user gets CSVs without Drive
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _build_download_cell(list(targets.keys())),
    })

    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(targets) if len(targets) <= 4 else f"{len(targets)}genes"
    preset_slug = re.sub(r"[^a-z0-9]+", "_", preset_name.lower()).strip("_")
    out_path = out_dir / f"perturb_{suffix}_{preset_slug}_{uuid.uuid4().hex[:6]}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path, targets


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["TXNDC15", "SYVN1", "MARCHF6"]
    nb_path, resolved = generate_notebook(syms)
    print("resolved:", resolved)
    print("wrote:", nb_path)
