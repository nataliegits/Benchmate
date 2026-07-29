"""Three more cross-check judges, all gene/variant level and all (nearly) free.

Each answers a *different* question than the LLM judge, so each is an independent
axis for the panel:

  Open Targets  — is this gene genuinely ASSOCIATED with the disease?
                  (genetics + literature + drugs)            free GraphQL, no key
  DepMap        — is this gene a real DEPENDENCY (essential) in the disease's
                  cancer cell lines?                          local CSV download
  AlphaMissense — is a coding variant likely PATHOGENIC?      free Ensembl VEP

All fail-soft and lazy: importing needs nothing, and any error returns None.

Each returns a scalar where **higher = more support** for the hypothesis, so it
plugs straight into benchmark/elo_vs_variant_score.correlate against the Elo
ranking, exactly like AlphaGenome and Boltz.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Open Targets keys diseases by MONDO/EFO id, but they vary — so we resolve the
# disease by NAME at query time instead of hardcoding an id.
DEFAULT_DISEASE = "multiple myeloma"
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
ENSEMBL_VEP = "https://rest.ensembl.org/vep/human/region"
_DEPMAP_DIR = Path(__file__).resolve().parent.parent / "data" / "depmap"
DEPMAP_CSV = Path(os.environ.get("DEPMAP_CSV", _DEPMAP_DIR / "CRISPRGeneEffect.csv"))
# Optional cell-line metadata. If present, we restrict the dependency score to the
# disease's own cell lines (e.g. multiple myeloma) instead of pan-cancer. Download
# Model.csv from the same DepMap data page. Without it we fall back to all lines.
DEPMAP_MODEL_CSV = Path(os.environ.get("DEPMAP_MODEL_CSV", _DEPMAP_DIR / "Model.csv"))
DEPMAP_LINEAGE = os.environ.get("DEPMAP_LINEAGE", "myeloma")  # substring, any column


# ---------------------------------------------------------------------------
# Open Targets — gene ↔ disease association  (free GraphQL, no key)
# ---------------------------------------------------------------------------

def opentargets_available() -> bool:
    try:
        import httpx  # noqa: F401
        return True
    except Exception:
        return False


def _ot_query(q: str, variables: dict) -> dict | None:
    import httpx
    with httpx.Client(timeout=30.0) as client:
        r = client.post(OT_URL, json={"query": q, "variables": variables})
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200 or data.get("errors"):
            # surface the GraphQL message instead of swallowing it
            print(f"[target_scorer] Open Targets {r.status_code}: "
                  f"{data.get('errors') or r.text[:300]}")
        return data


def _resolve(query_string: str, entity: str) -> str | None:
    q = ('query($s:String!,$e:[String!]){ search(queryString:$s, entityNames:$e) '
         '{ hits { id name entity } } }')
    data = _ot_query(q, {"s": query_string, "e": [entity]})
    hits = (((data or {}).get("data") or {}).get("search") or {}).get("hits") or []
    return hits[0]["id"] if hits else None


def opentargets_score(symbol: str, disease: str = DEFAULT_DISEASE) -> float | None:
    """Overall Open Targets association (0–1) between a gene and a disease,
    resolving both by name/symbol. Higher = stronger evidence the gene is linked
    to the disease. 0.0 = resolved but no association on record; None on error."""
    if not opentargets_available():
        return None
    try:
        ens = _resolve(symbol, "target")
        did = _resolve(disease, "disease")
        if not ens or not did:
            return None
        # `Bs` filters associatedDiseases to specific disease (B-side) IDs.
        q = ('query($e:String!,$d:[String!]){ target(ensemblId:$e){ '
             'associatedDiseases(Bs:$d){ rows { score disease { id } } } } }')
        data = _ot_query(q, {"e": ens, "d": [did]})
        rows = ((((data or {}).get("data") or {}).get("target") or {})
                .get("associatedDiseases") or {}).get("rows") or []
        for row in rows:
            if (row.get("disease") or {}).get("id") == did:
                return float(row["score"])
        return 0.0   # resolved cleanly, no association on record for this disease
    except Exception as e:
        print(f"[target_scorer] Open Targets error for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# DepMap — gene dependency / essentiality  (from the public CRISPR CSV)
# ---------------------------------------------------------------------------
# DepMap has no clean free API; the authoritative source is the downloadable
# CRISPR gene-effect matrix. Download `CRISPRGeneEffect.csv` from
# https://depmap.org/portal/data_page/ into data/depmap/ (or set DEPMAP_CSV).
# Gene effect is negative when a gene is essential, so we return -mean(effect):
# higher = more of a dependency. If Model.csv is also present we average only over
# the disease's own cell lines (multiple myeloma) — a more on-target dependency
# than the pan-cancer mean.

DEPMAP_SUMMARY = Path(os.environ.get(
    "DEPMAP_SUMMARY", _DEPMAP_DIR / "gene_effect_summary.csv"))


def depmap_available() -> bool:
    """True if either the full matrix or the committed summary is present.

    The full CSV is 440 MB and gitignored, so on a hosted deploy only the
    summary exists. Requiring the big file meant DepMap was permanently
    unavailable to anyone who didn't run Benchmate locally.
    """
    try:
        import pandas  # noqa: F401
        return DEPMAP_CSV.exists() or DEPMAP_SUMMARY.exists()
    except Exception:
        return False


def depmap_source() -> str:
    """Which file a score will come from — 'full', 'summary', or 'none'.
    The UI shows this so a number's provenance is never ambiguous."""
    if DEPMAP_CSV.exists():
        return "full"
    if DEPMAP_SUMMARY.exists():
        return "summary"
    return "none"


@lru_cache(maxsize=1)
def _depmap_summary_frame():
    """The small per-gene summary, indexed by symbol."""
    import pandas as pd
    df = pd.read_csv(DEPMAP_SUMMARY)
    df["gene"] = df["gene"].astype(str).str.upper()
    return df.set_index("gene")


def depmap_lineage_in_use() -> str:
    """The lineage the numbers actually reflect. Reads it from the summary
    rather than assuming, because a pan-cancer average and a myeloma-restricted
    one are different claims."""
    try:
        if DEPMAP_CSV.exists():
            return DEPMAP_LINEAGE if _myeloma_model_ids() else "all"
        return str(_depmap_summary_frame()["lineage"].iloc[0])
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def _depmap_frame():
    """The CRISPR gene-effect matrix, loaded once per process.

    CRISPRGeneEffect.csv is ~440 MB. Scoring a four-gene hypothesis used to
    re-parse the whole thing four times, which is tens of seconds of dead air
    per click. Cached so the cost is paid once.
    """
    import pandas as pd
    return pd.read_csv(DEPMAP_CSV, index_col=0)


def _myeloma_model_ids() -> set[str] | None:
    """ModelIDs whose metadata mentions the target lineage (default 'myeloma').
    Returns None if Model.csv is absent or no rows match — caller falls back to
    all cell lines. Matches the substring in ANY column, so it's robust to which
    OncoTree column DepMap uses."""
    if not DEPMAP_MODEL_CSV.exists():
        return None
    try:
        import pandas as pd
        m = pd.read_csv(DEPMAP_MODEL_CSV, dtype=str).fillna("")
        id_col = "ModelID" if "ModelID" in m.columns else m.columns[0]
        needle = DEPMAP_LINEAGE.lower()
        text = m.drop(columns=[id_col], errors="ignore").apply(
            lambda r: " ".join(r.values).lower(), axis=1)
        ids = set(m.loc[text.str.contains(needle), id_col])
        return ids or None
    except Exception as e:
        print(f"[target_scorer] DepMap Model.csv read failed ({e}); using all lines")
        return None


def depmap_score(symbol: str) -> float | None:
    """Mean dependency for a gene: -mean(gene effect). Higher = more essential.
    Restricted to multiple-myeloma cell lines when data/depmap/Model.csv is
    present, else averaged over all lines. None if the CSV is missing or the gene
    isn't found."""
    if not depmap_available():
        return None

    # No full matrix (the usual case on a hosted deploy) — use the committed
    # per-gene summary. Same underlying DepMap numbers, precomputed.
    if not DEPMAP_CSV.exists():
        try:
            row = _depmap_summary_frame().loc[symbol.upper()]
            return float(-float(row["mean_effect"]))
        except Exception:
            return None

    try:
        df = _depmap_frame()
        # columns look like "SYVN1 (ENSG...)" — match on the leading symbol
        col = next((c for c in df.columns if c.split(" ")[0].upper() == symbol.upper()),
                   None)
        if col is None:
            return None
        series = df[col]
        ids = _myeloma_model_ids()
        if ids:
            subset = series[series.index.isin(ids)]
            if len(subset) >= 3:          # enough lines to mean meaningfully
                series = subset
        return float(-series.mean())
    except Exception as e:
        print(f"[target_scorer] DepMap error for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# AlphaMissense — coding-variant pathogenicity  (free, via Ensembl VEP)
# ---------------------------------------------------------------------------

def alphamissense_available() -> bool:
    try:
        import httpx  # noqa: F401
        return True
    except Exception:
        return False


def alphamissense_score(chrom: str, pos: int, ref: str, alt: str) -> float | None:
    """AlphaMissense pathogenicity (0–1) for a coding (missense) variant, via the
    Ensembl VEP REST API. Higher = more likely pathogenic. None on error / if the
    variant isn't missense. VERIFY the response shape against rest.ensembl.org.

    region format Ensembl expects: e.g. "9:136219502-136219502:1" + allele "T".
    """
    if not alphamissense_available():
        return None
    try:
        import httpx
        c = str(chrom).replace("chr", "")
        region = f"{c}:{pos}-{pos}:1"
        url = f"{ENSEMBL_VEP}/{region}/{alt}"
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, params={"AlphaMissense": 1},
                           headers={"Content-Type": "application/json"})
            r.raise_for_status()
            data = r.json()
        best = None
        for entry in data:
            for tc in entry.get("transcript_consequences", []):
                am = tc.get("alphamissense") or {}
                v = am.get("am_pathogenicity")
                if isinstance(v, (int, float)):
                    best = max(best, float(v)) if best is not None else float(v)
        return best
    except Exception as e:
        print(f"[target_scorer] AlphaMissense error: {e}")
        return None


if __name__ == "__main__":
    print("Open Targets:", opentargets_available(),
          "| DepMap CSV:", depmap_available(), f"({DEPMAP_CSV})",
          "| AlphaMissense (VEP):", alphamissense_available())
    if opentargets_available():
        print("SYVN1 vs multiple myeloma:", opentargets_score("SYVN1"))
        print("OR2T1 (control) vs multiple myeloma:", opentargets_score("OR2T1"))
