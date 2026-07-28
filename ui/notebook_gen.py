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
        "4. Aggregates each perturbation into a per-gene result table.\n"
        "intersections.\n",
        "5. Writes one {GENE}_stats.csv per target and downloads it for Benchmate.\n"
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
    """Final cell: normalise each perturbation result into a Benchmate-ready
    CSV and download it.

    Benchmate's cache is strict about two things — the filename must be
    `{SYMBOL}_stats.csv` (that IS the index) and the columns must include
    Affected_Ensembl_ID / Affected_gene_name / Cosine_sim_mean / N_Detections.
    So rather than trusting the path, we locate whatever the stats step wrote,
    check the columns, write a clean copy, and say plainly whether each gene is
    good to upload.
    """
    return [
        "# ---- Export for Benchmate -------------------------------------------\n",
        "# Writes one {GENE}_stats.csv per target, checks it has the columns\n",
        "# Benchmate needs, and downloads it to your machine.\n",
        "import os, glob, shutil\n",
        "import pandas as pd\n",
        "\n",
        f"DOWNLOAD_TARGETS = {target_symbols}\n",
        "EXPORT_DIR = '/content/benchmate_export'\n",
        "os.makedirs(EXPORT_DIR, exist_ok=True)\n",
        "\n",
        "# symbol lookup, in case the stats file only carries Ensembl IDs\n",
        "try:\n",
        "    ens_to_sym = dict(zip(adata.var['feature_id'], adata.var['feature_name']))\n",
        "except Exception:\n",
        "    ens_to_sym = {}\n",
        "\n",
        "REQUIRED = ['Affected_Ensembl_ID', 'Cosine_sim_mean']\n",
        "NICE     = ['Affected_gene_name', 'N_Detections']\n",
        "\n",
        "def find_stats_csv(sym):\n",
        "    \"\"\"Exact path first, then glob — the stats step names files itself.\"\"\"\n",
        "    exact = f'{PERTURB_OUT}/{sym}/{sym}_stats.csv'\n",
        "    if os.path.exists(exact):\n",
        "        return exact\n",
        "    hits = glob.glob(f'{PERTURB_OUT}/{sym}/*.csv')\n",
        "    return hits[0] if hits else None\n",
        "\n",
        "ready, problems = [], []\n",
        "for sym in DOWNLOAD_TARGETS:\n",
        "    src = find_stats_csv(sym)\n",
        "    if not src:\n",
        "        problems.append((sym, 'no CSV found — did the perturbation finish?'))\n",
        "        continue\n",
        "    df = pd.read_csv(src)\n",
        "    if 'Affected' in df.columns:\n",
        "        df = df[df['Affected'] != 'cell_emb']\n",
        "    missing = [c for c in REQUIRED if c not in df.columns]\n",
        "    if missing:\n",
        "        problems.append((sym, f'missing columns {missing}'))\n",
        "        continue\n",
        "    # add gene symbols if the stats file only has Ensembl IDs\n",
        "    if 'Affected_gene_name' not in df.columns:\n",
        "        try:\n",
        "            df['Affected_gene_name'] = df['Affected_Ensembl_ID'].map(ens_to_sym)\n",
        "        except Exception:\n",
        "            df['Affected_gene_name'] = df['Affected_Ensembl_ID']\n",
        "    if 'N_Detections' not in df.columns:\n",
        "        df['N_Detections'] = 0\n",
        "    df = df.dropna(subset=['Affected_Ensembl_ID'])\n",
        "    df = df.sort_values('Cosine_sim_mean', ascending=True)\n",
        "    out = f'{EXPORT_DIR}/{sym}_stats.csv'\n",
        "    df.to_csv(out, index=False)\n",
        "    top = df.iloc[0]['Affected_gene_name'] if len(df) else '-'\n",
        "    ready.append((sym, out, len(df), top))\n",
        "\n",
        "print('Ready for Benchmate')\n",
        "print('-' * 58)\n",
        "for sym, out, n, top in ready:\n",
        "    print(f'  {sym:>10}_stats.csv   {n:>5} rows   top hit: {top}')\n",
        "for sym, why in problems:\n",
        "    print(f'  {sym:>10}  SKIPPED — {why}')\n",
        "\n",
        "# keep a copy on Drive too, if it happens to be mounted\n",
        "try:\n",
        "    if os.path.isdir('/content/drive/MyDrive'):\n",
        "        dst = '/content/drive/MyDrive/benchmate_export'\n",
        "        os.makedirs(dst, exist_ok=True)\n",
        "        for _, out, _, _ in ready:\n",
        "            shutil.copy(out, dst)\n",
        "        print(f'\\nAlso copied to {dst}')\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "# download each one (Colab sometimes drops rapid-fire downloads, so pause)\n",
        "try:\n",
        "    from google.colab import files\n",
        "    import time\n",
        "    for _, out, _, _ in ready:\n",
        "        files.download(out)\n",
        "        time.sleep(1.5)\n",
        "    print('\\nDownloads triggered — check your Downloads folder.')\n",
        "except Exception as e:\n",
        "    print(f'\\n(Auto-download unavailable: {e})')\n",
        "    print(f'Grab them from the file browser at {EXPORT_DIR}')\n",
        "\n",
        "print('\\nNext: open Benchmate -> sidebar -> Upload CSVs, and drop these in.')\n",
        "print('The filename IS the index, so keep the {GENE}_stats.csv names as-is.')\n",
    ]


# ---------------------------------------------------------------------------
# Dependency install — GENERATED, never taken from the template.
# ---------------------------------------------------------------------------
# Hardwired here on purpose. Geneformer + transformers 4.40 are from the
# numpy<2 era while Colab ships numpy 2.x, so installing over a live session
# leaves a half-swapped numpy (new Python files, old compiled
# _multiarray_umath) and the notebook dies later with:
#     AttributeError: module 'numpy._core._multiarray_umath'
#                     has no attribute '_blas_supports_fpe'
# Three things prevent it: ONE pip transaction, numpy pinned, and an automatic
# runtime restart. Anyone downloading a notebook from Benchmate gets this —
# it does not depend on the template file being correct.

def _build_install_cell() -> list[str]:
    return [
        "# Dependency install — run this FIRST, once per session.\n",
        "#\n",
        "# The golden rule on Colab: NEVER let pip move numpy. Colab's stack\n",
        "# (pandas, scipy, sklearn...) is compiled against whatever numpy ships in\n",
        "# the image. Change numpy and you get one of these two failures:\n",
        "#   * downgrade -> ValueError: numpy.dtype size changed ... Expected 96\n",
        "#                  from C header, got 88 from PyObject\n",
        "#   * half-swap -> AttributeError: '_multiarray_umath' has no attribute\n",
        "#                  '_blas_supports_fpe'\n",
        "# So we pin numpy to the version ALREADY installed via a pip constraints\n",
        "# file, and let pip pick versions of everything else that fit around it.\n",
        "\n",
        "def _deps_ready():\n",
        "    try:\n",
        "        import numpy, pandas, geneformer, cellxgene_census  # noqa: F401\n",
        "        return True\n",
        "    except Exception:\n",
        "        return False\n",
        "\n",
        "if _deps_ready():\n",
        "    print('deps already installed — nothing to do.')\n",
        "else:\n",
        "    import numpy, subprocess, sys\n",
        "    # freeze numpy at the version Colab already has\n",
        "    with open('/tmp/constraints.txt', 'w') as f:\n",
        "        f.write(f'numpy=={numpy.__version__}\\n')\n",
        "    print(f'holding numpy at {numpy.__version__}')\n",
        "    # transformers must stay on 4.x: Geneformer does\n",
        "    #   from transformers import SpecialTokensMixin\n",
        "    # which v5 removed from the top-level namespace. 4.44+ is also\n",
        "    # numpy-2 compatible, so this plays nicely with the pin above.\n",
        "    pkgs = ['transformers>=4.44,<5', 'tokenizers', 'peft',\n",
        "            'accelerate', 'datasets<4', 'cellxgene-census',\n",
        "            'loompy', 'mygene', 'tdigest', 'anndata', 'pyarrow',\n",
        "            'seaborn', 'statsmodels', 'optuna']\n",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n",
        "                    '--no-cache-dir', '-c', '/tmp/constraints.txt', *pkgs],\n",
        "                   check=False)\n",
        "    # Geneformer with --no-deps so it can't drag numpy or torch around\n",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n",
        "                    '--no-cache-dir', '--no-deps',\n",
        "                    'git+https://huggingface.co/ctheodoris/Geneformer'],\n",
        "                   check=False)\n",
        "    print('Installed. RESTARTING the runtime for a clean import state —')\n",
        "    print('Colab will say the session crashed. That is expected.')\n",
        "    print('When it returns, skip this cell and run the next one.')\n",
        "    import IPython\n",
        "    IPython.Application.instance().kernel.do_shutdown(True)\n",
    ]


def _build_verify_cell() -> list[str]:
    """A fast smoke test so a broken environment fails HERE with a clear
    message, instead of five cells later inside someone else's traceback."""
    return [
        "# Sanity check — confirms numpy wasn't disturbed and the stack imports.\n",
        "#\n",
        "# Also disables one landmine: HuggingFace `datasets` does\n",
        "#     from torchvision.io import VideoReader\n",
        "# when it thinks torchvision is present, and current torchvision no\n",
        "# longer exports VideoReader — which blows up mid-perturbation. We\n",
        "# never touch video, so switch that path off.\n",
        "try:\n",
        "    import datasets.config\n",
        "    datasets.config.TORCHVISION_AVAILABLE = False\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "import numpy, pandas\n",
        "print('numpy  ', numpy.__version__)\n",
        "print('pandas ', pandas.__version__)\n",
        "try:\n",
        "    numpy.zeros(3) + pandas.Series([1, 2, 3]).to_numpy()\n",
        "    import transformers\n",
        "    from transformers import SpecialTokensMixin  # v5 removed this\n",
        "    from geneformer import TranscriptomeTokenizer  # the real test\n",
        "    import cellxgene_census, mygene  # noqa: F401\n",
        "    print('transformers', transformers.__version__)\n",
        "    print('OK — environment is consistent, carry on.')\n",
        "except Exception as e:\n",
        "    print('PROBLEM:', type(e).__name__, e)\n",
        "    print('Fix: Runtime > Disconnect and delete runtime, then run the\\n'\n",
        "          'setup cell again from a fresh session.')\n",
    ]


def _is_install_cell(src: str) -> bool:
    """A pip cell that touches the heavy ML stack (not a stray one-liner)."""
    return ("pip" in src and any(k in src.lower() for k in
            ("geneformer", "transformers", "cellxgene")))


def _neutralise_stray_installs(nb: dict) -> int:
    """Later `!pip install -q ...` cells can re-resolve numpy mid-run and undo
    the pin. Everything they ask for is already in the install cell, so replace
    them with a no-op note. Returns how many were neutralised."""
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "pip install" in src and not _is_install_cell(src):
            cell["source"] = [
                "# (installs handled in the setup cell at the top — nothing to do here,\n",
                "#  re-running pip mid-session can unpin numpy and break Geneformer)\n",
            ]
            cell["outputs"] = []
            cell["execution_count"] = None
            n += 1
    return n


def _despecialise(nb: dict, targets: dict, preset_name: str, preset: dict) -> None:
    """Strip the template's original study (ciliated cells / TXNDC15 / CiliaCarta)
    out of every remaining cell.

    The template was written for one specific experiment. Only three cells used
    to be rewritten, so a generated notebook still said things like "Pull
    ciliated cell types" and shipped a hard-coded CiliaCarta gene list — noise
    for anyone studying something else. Here we rewrite the section headings to
    plainly describe what each section DOES, retarget the paths, and generalise
    the comparison cells to the user's own genes.
    """
    import re as _re
    syms = list(targets)
    primary = syms[0] if syms else "your gene"
    others = syms[1:]
    slug = _re.sub(r"[^a-z0-9]+", "_", preset_name.lower()).strip("_") or "run"

    md = {
        "CELLxGENE Census": (
            "## 2. Pull cells from CELLxGENE Census\n\n"
            f"Downloads a single-cell expression sample for the chosen cell "
            f"context (**{preset_name}**) and keeps the genes Geneformer needs.\n"),
        "## 4. Perturbation": (
            "## 4. In-silico perturbation\n\n"
            "Deletes each target gene in the model and measures how far every "
            "other gene's embedding moves as a result.\n"),
        "Stats + intersection": (
            "## 5. Top affected genes per target\n\n"
            "For each gene you perturbed, the genes whose embeddings moved most "
            "— your ranked shortlist.\n"),
        "Enrichment + explicit cilium overlap": (
            "## 6. Pathway enrichment\n\n"
            "Runs the top affected genes through Enrichr (GO / Reactome / KEGG), "
            "then optionally checks overlap with a gene set you supply.\n"),
        "## What to look for": (
            "## What to look for\n\n"
            f"- **Enrichment terms that match your hypothesis** for {primary} — "
            "the pathway you expected shows up near the top.\n"
            "- **A gene-specific signature** — affected genes unique to one "
            "target, not shared by the others, point to a distinct mechanism.\n"
            "- **A shared signature** across all targets usually means a common "
            "pathway (or a batch/tissue artefact — check the controls).\n"
            "- **Nothing enriched** is a real result too: the model may not "
            "represent this biology in the chosen cell context.\n"),
    }
    # The first markdown cell is the title/intro — the generator writes it via
    # _build_intro_md, so never let the section rules below overwrite it.
    intro_i = next((i for i, c in enumerate(nb["cells"])
                    if c.get("cell_type") == "markdown"), None)
    if intro_i is not None:
        isrc = "".join(nb["cells"][intro_i].get("source", []))
        if _re.search(r"cilia|ciliat|axoneme|ependymal|hepatocyte|TXNDC15",
                      isrc, _re.I):
            nb["cells"][intro_i]["source"] = [
                "# Geneformer in-silico perturbation\n\n",
                f"**Targets:** {', '.join(syms)}  \n",
                f"**Cell context:** {preset_name}\n\n",
                "Deletes each gene in the model and reports which other genes "
                "shift the most, then runs pathway enrichment on the result.\n",
            ]

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "markdown" or i == intro_i:
            continue
        src = "".join(cell.get("source", []))
        for needle, replacement in md.items():
            if needle in src:
                cell["source"] = [replacement]
                break

    # Safety net: if any markdown still mentions the template's original study,
    # keep the heading and drop the stale body rather than shipping it.
    STALE = _re.compile(r"cilia|ciliat|ciliopath|axoneme|CiliaCarta|ependymal"
                        r"|hepatocyte|TXNDC15", _re.I)
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "markdown" or i == intro_i:
            continue
        src = "".join(cell.get("source", []))
        if not STALE.search(src):
            continue
        heading = next((ln for ln in src.splitlines() if ln.startswith("#")), "")
        keep = [ln for ln in src.splitlines()
                if ln.strip() and not STALE.search(ln) and not ln.startswith("#")]
        body = ("\n".join(keep[:2]) + "\n") if keep else ""
        cell["source"] = [f"{heading}\n\n" if heading else "", body] if heading \
            else [body or "\n"]

    # --- code: retarget paths that carry the old study's name ---
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "benchmate_geneformer_cilia" in src or "/content/cilia_" in src:
            src = src.replace("benchmate_geneformer_cilia",
                              f"benchmate_geneformer_{slug}")
            src = src.replace("/content/cilia_", "/content/gf_")
            cell["source"] = src.splitlines(keepends=True)

    # --- code: guard the perturbation cell against the torchvision/datasets clash
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "InSilicoPerturber" in src and "datasets.config" not in src:
            cell["source"] = [
                "# `datasets` imports torchvision's VideoReader when it thinks\n",
                "# torchvision is installed; current torchvision dropped it. No video\n",
                "# here, so turn that path off before perturbing.\n",
                "try:\n",
                "    import datasets.config\n",
                "    datasets.config.TORCHVISION_AVAILABLE = False\n",
                "except Exception:\n",
                "    pass\n",
                "\n",
            ] + list(src.splitlines(keepends=True))

    # --- code: stale TARGET_SYMBOLS list left in the preamble ---
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "TARGET_SYMBOLS" in src and "TARGETS" in src:
            src = _re.sub(r"^\s*TARGET_SYMBOLS\s*=\s*\[[^\]]*\]\s*\n", "",
                          src, flags=_re.M)
            cell["source"] = src.splitlines(keepends=True)

    # --- code: the tokenised copy was named after the old study ---
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "cilia" in src.lower():
            # the tokeniser names its output after this prefix
            src = (src.replace("cilia.h5ad", "data.h5ad")
                      .replace('"cilia"', '"data"')
                      .replace("cilia.dataset", "data.dataset"))
            cell["source"] = src.splitlines(keepends=True)

    # --- code: hard-coded 3-way intersection -> loop over the real targets ---
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "shared3" not in src:
            continue
        head = src.split("shared3")[0].rstrip() + "\n\n"
        cell["source"] = list(head.splitlines(keepends=True)) + [
            "# Top affected genes for each target — the shortlist you actually use.\n",
            "for sym in TARGETS:\n",
            "    df = tops[sym].head(15)\n",
            "    print(f\"\\n=== {sym}: top {len(df)} affected genes ===\")\n",
            "    for _, r in df.iterrows():\n",
            "        shift = 1 - r['Cosine_sim_mean']\n",
            "        print(f\"  {str(r.get('Affected_gene_name','?')):>12}  \"\n",
            "              f\"shift={shift:.4f}  N={int(r.get('N_Detections', 0))}\")\n",
        ]
        cell["outputs"] = []
        cell["execution_count"] = None

    # --- code: enrich() calls naming the old genes ---
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "pair_txn_syvn" in src or "pair_txn_mar" in src:
            src = _re.sub(r"^.*pair_txn_\w+.*\n", "", src, flags=_re.M)
            cell["source"] = src.splitlines(keepends=True)

    # --- code: the CiliaCarta block -> an optional user gene set ---
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if "CILIA_GENES" not in "".join(cell.get("source", [])):
            continue
        cell["source"] = [
            "# OPTIONAL: overlap with a gene set of your own.\n",
            "# Put the symbols you care about here (a pathway, a screen hit list,\n",
            "# a curated set) and each target is scored against it. Leave empty to skip.\n",
            "MY_GENE_SET = set()   # e.g. {\"SEL1L\", \"EDEM1\", \"OS9\", \"DERL1\"}\n",
            "\n",
            "def gene_set_overlap(ens_ids, label):\n",
            "    syms = {ens_to_sym.get(g, g) for g in ens_ids}\n",
            "    hit = syms & MY_GENE_SET\n",
            "    pct = 100 * len(hit) / max(len(syms), 1)\n",
            "    print(f\"{label:>20}  : {len(hit):>3}/{len(syms)} overlap \"\n",
            "          f\"({pct:.1f}%) -- {sorted(hit)}\")\n",
            "    return hit\n",
            "\n",
            "if MY_GENE_SET:\n",
            "    print('=== overlap with MY_GENE_SET ===')\n",
            "    for sym in TARGETS:\n",
            "        gene_set_overlap(tops[sym]['Affected_Ensembl_ID'], sym)\n",
            "else:\n",
            "    print('MY_GENE_SET is empty — skipping. Add symbols above to use this.')\n",
        ]
        cell["outputs"] = []
        cell["execution_count"] = None

    # --- code: the TXNDC15-vs-others comparison -> the user's own targets ---
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "txn_only" not in src and "TXNDC15_only" not in src:
            continue
        if len(syms) < 2:
            cell["source"] = [
                "# (gene-specific comparison needs 2+ targets — only one here)\n"]
        else:
            cell["source"] = [
                "# What is each target doing that the others are NOT?\n",
                "# Set-subtracts the other targets' affected genes, then enriches\n",
                "# whatever is left — that residue is the gene-specific signature.\n",
                "for sym in TARGETS:\n",
                "    mine = set(tops[sym]['Affected_Ensembl_ID'])\n",
                "    for other in TARGETS:\n",
                "        if other != sym:\n",
                "            mine -= set(tops[other]['Affected_Ensembl_ID'])\n",
                "    print(f'\\n=== {sym}-specific: {len(mine)} genes ===')\n",
                "    if mine:\n",
                "        enrich(list(mine), f'{sym}_only')\n",
            ]
        cell["outputs"] = []
        cell["execution_count"] = None


    # --- drop the downstream analysis sections -------------------------------
    # Benchmate does the interpretation. This notebook's only job is: run the
    # perturbation, write the CSVs, hand them over. Per-gene printouts, pathway
    # enrichment and "follow up" ideas just make it longer and give people more
    # to debug, so they go.
    DROP_CODE = ("load_stats", "import gseapy", "MY_GENE_SET", "CILIA_GENES",
                 "top {len(df)} affected genes", "gene-specific signature",
                 "shared_all", "txn_only", "cilium_overlap", "gene_set_overlap")
    DROP_MD = ("## 6. Pathway enrichment", "Section 7", "## What to look for",
               "## 5. Top affected genes", "Enrichment", "Follow Up")
    keep = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and any(m in src for m in DROP_CODE):
            continue
        if cell.get("cell_type") == "markdown" and any(m in src for m in DROP_MD):
            continue
        keep.append(cell)
    nb["cells"] = keep

    # the stats step is what WRITES the CSVs, so it stays — relabel it
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code" and \
                "InSilicoPerturberStats" in "".join(cell.get("source", [])):
            if i > 0 and nb["cells"][i - 1].get("cell_type") == "markdown":
                nb["cells"][i - 1]["source"] = [
                    "## 5. Write the per-gene result files\n\n"
                    "Aggregates each perturbation into a `{GENE}_stats.csv`.\n"]
            break



    # Remove dead cells rather than leaving no-op placeholders behind.
    nb["cells"] = [
        c for c in nb["cells"]
        if not (c.get("cell_type") == "code"
                and "".join(c.get("source", [])).strip().startswith(
                    "# (installs handled in the setup cell"))
        and not (c.get("cell_type") == "code"
                 and "".join(c.get("source", [])).strip().startswith("!pip show"))
    ]

    # Renumber the export heading so it follows the last real section.
    nums = [int(mm.group(1)) for c in nb["cells"]
            if c.get("cell_type") == "markdown"
            for mm in [_re.match(r"##\s*(\d+)\.", "".join(c.get("source", [])))]
            if mm and "Export" not in "".join(c.get("source", []))]
    nxt = (max(nums) + 1) if nums else 5
    for cell in nb["cells"]:
        if cell.get("cell_type") == "markdown" and \
                "Export for Benchmate" in "".join(cell.get("source", [])):
            cell["source"] = [
                f"## {nxt}. Export for Benchmate\n\n"
                "Writes one `{GENE}_stats.csv` per target, checks it has the "
                "columns Benchmate needs, and downloads it. Upload these in "
                "Benchmate's sidebar under **Upload CSVs**.\n"]


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
    # Force the install cell in, whatever the template says. If the template
    # has one we overwrite it; if it doesn't, we insert one at the top.
    install_src = _build_install_cell()
    install_done = False
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and _is_install_cell(
                "".join(cell.get("source", []))):
            cell["source"] = install_src
            cell["outputs"] = []
            cell["execution_count"] = None
            install_done = True
            break
    if not install_done:
        first_code = next((i for i, c in enumerate(nb["cells"])
                           if c.get("cell_type") == "code"), 0)
        nb["cells"].insert(first_code, {
            "cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": install_src})
    _neutralise_stray_installs(nb)

    # Verify cell: REPLACE any existing one (a template copy may be stale),
    # otherwise insert a fresh one right after the install cell.
    verify_src = _build_verify_cell()
    existing = next((c for c in nb["cells"]
                     if c.get("cell_type") == "code"
                     and "environment is consistent" in "".join(c.get("source", []))),
                    None)
    if existing is not None:
        existing["source"] = verify_src
        existing["outputs"] = []
        existing["execution_count"] = None
    else:
        idx = next((i for i, c in enumerate(nb["cells"])
                    if c.get("cell_type") == "code"
                    and "_deps_ready" in "".join(c.get("source", []))), 0)
        nb["cells"].insert(idx + 1, {
            "cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": verify_src})

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

    _despecialise(nb, targets, preset_name, preset)

    if not replaced["intro"]:
        raise RuntimeError("Template has no markdown cell to use as intro.")
    if not replaced["cellxgene"]:
        raise RuntimeError("Template lacks a CELLxGENE pull cell.")
    if not replaced["targets"]:
        raise RuntimeError("Template lacks a TARGETS = {...} cell.")

    # Append the Benchmate export section (heading + cell)
    # number this section after whatever the last real one turned out to be
    _nums = [int(_m.group(1)) for _c in nb["cells"]
             if _c.get("cell_type") == "markdown"
             for _m in [re.match(r"##\s*(\d+)\.", "".join(_c.get("source", [])))]
             if _m]
    _next = (max(_nums) + 1) if _nums else 5
    nb["cells"].append({
        "cell_type": "markdown", "metadata": {},
        "source": [f"## {_next}. Export for Benchmate\n\n"
                   "Writes one `{GENE}_stats.csv` per target, verifies it has "
                   "the columns Benchmate needs, and downloads it. Upload these "
                   "in Benchmate's sidebar under **Upload CSVs**.\n"],
    })
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
