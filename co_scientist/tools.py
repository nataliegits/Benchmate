"""External tools the agents can call.

Real:
    - PubMed search (NCBI E-utilities, no key required)
    - Geneformer perturbation lookup (reads pre-computed CSVs)

Stubbed:
    - BioNeMo NIM call (uncomment when you have an endpoint)
    - Embedding (uses a cheap hash proxy until you wire a real model)
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx


# ============================================================
# PubMed literature search
# ============================================================

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")
NCBI_KEY = os.environ.get("NCBI_API_KEY", "")


@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str
    year: str | None = None

    def short(self, n: int = 280) -> str:
        a = self.abstract[:n] + ("…" if len(self.abstract) > n else "")
        return f"[PMID {self.pmid}] {self.title}\n{a}"


def _params(**extra) -> dict:
    p = {"tool": "co-scientist-starter", "email": NCBI_EMAIL}
    if NCBI_KEY:
        p["api_key"] = NCBI_KEY
    p.update(extra)
    return p


def pubmed_search(query: str, max_results: int = 5, timeout: float = 15.0
                  ) -> list[Paper]:
    """Search PubMed and return parsed abstracts."""
    with httpx.Client(timeout=timeout) as client:
        # 1. ESearch -> list of PMIDs
        r = client.get(f"{NCBI_BASE}/esearch.fcgi", params=_params(
            db="pubmed", term=query, retmax=max_results, retmode="json"))
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # 2. EFetch -> XML with titles + abstracts
        r = client.get(f"{NCBI_BASE}/efetch.fcgi", params=_params(
            db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml"))
        r.raise_for_status()
        xml = r.text

    # very-light XML parsing; for production use lxml or biopython
    papers: list[Paper] = []
    for block in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, re.DOTALL):
        pmid = re.search(r"<PMID[^>]*>(\d+)</PMID>", block)
        title = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, re.DOTALL)
        abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>",
                                    block, re.DOTALL)
        year = re.search(r"<PubDate>.*?<Year>(\d+)</Year>", block, re.DOTALL)
        if pmid and title:
            clean = lambda s: re.sub(r"<[^>]+>", "", s).strip()
            papers.append(Paper(
                pmid=pmid.group(1),
                title=clean(title.group(1)),
                abstract=clean(" ".join(abstract_parts)),
                year=year.group(1) if year else None,
            ))
    return papers


# ============================================================
# Embedding (PLACEHOLDER — replace with a real model)
# ============================================================

def embed(text: str, dim: int = 64) -> list[float]:
    """Cheap deterministic embedding so the Proximity agent has *something*
    to cluster on until you swap in voyage-3 / text-embedding-3-large /
    a BioNeMo ESM-2 endpoint for protein-specific clustering.
    """
    h = hashlib.sha256(text.encode()).digest()
    return [(h[i % len(h)] - 128) / 128.0 for i in range(dim)]


def cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


# ============================================================
# Geneformer in-silico perturbation lookup
# ============================================================
#
# Reads pre-computed perturbation result CSVs produced by notebook 02.
# Place each gene's *_stats.csv in data/geneformer/ at the repo root.
# The lookup is fast (just a pandas read + filter), so the Generation and
# Reflection agents can call it inline during the supervisor loop.

GENEFORMER_DIR = Path(__file__).resolve().parent.parent / "data" / "geneformer"


def available_geneformer_genes() -> list[str]:
    """List gene symbols with cached perturbation results."""
    if not GENEFORMER_DIR.exists():
        return []
    return sorted(p.stem.replace("_stats", "")
                  for p in GENEFORMER_DIR.glob("*_stats.csv"))


def geneformer_neighbors(gene_symbol: str, top_n: int = 20,
                         min_detections: int = 200) -> dict:
    """Top-N genes whose embedding shifted most when `gene_symbol` was deleted.

    Returns a dict with keys:
        gene_symbol     — query gene
        n_results       — number of affected genes returned
        affected_genes  — list of {symbol, ensembl_id, cosine_shift, n_detections}
    or {"error": "..."} if the gene isn't in the cache.

    cosine_shift = 1 - cosine_sim_mean, so larger = bigger predicted effect.
    """
    path = GENEFORMER_DIR / f"{gene_symbol}_stats.csv"
    if not path.exists():
        return {"error": f"No cached Geneformer results for {gene_symbol}. "
                         f"Available: {available_geneformer_genes()}"}

    try:
        import pandas as pd
    except ImportError:
        return {"error": "pandas not installed; pip install pandas"}

    df = pd.read_csv(path)
    if "Affected" in df.columns:
        df = df[df["Affected"] != "cell_emb"]
    df = df.dropna(subset=["Affected_Ensembl_ID"])
    if "N_Detections" in df.columns:
        df = df[df["N_Detections"] >= min_detections]
    df = df.sort_values("Cosine_sim_mean", ascending=True).head(top_n)

    return {
        "gene_symbol": gene_symbol,
        "n_results": len(df),
        "affected_genes": [
            {"symbol": row.get("Affected_gene_name"),
             "ensembl_id": row["Affected_Ensembl_ID"],
             "cosine_shift": round(1 - row["Cosine_sim_mean"], 4),
             "n_detections": int(row.get("N_Detections", 0))}
            for _, row in df.iterrows()
        ],
    }


def geneformer_intersect(gene_symbols: list[str], top_n: int = 200,
                         min_detections: int = 200) -> dict:
    """Genes affected under ALL of the listed perturbations.

    Returns {perturbations, n_shared, shared_genes:[{symbol, ensembl_id}]}.
    """
    sets: dict[str, dict[str, dict]] = {}
    for g in gene_symbols:
        r = geneformer_neighbors(g, top_n=top_n, min_detections=min_detections)
        if "error" in r:
            return r
        sets[g] = {ag["ensembl_id"]: ag for ag in r["affected_genes"]}

    if not sets:
        return {"perturbations": gene_symbols, "n_shared": 0, "shared_genes": []}

    shared_ids = set.intersection(*[set(s.keys()) for s in sets.values()])
    first = next(iter(sets.values()))
    return {
        "perturbations": gene_symbols,
        "n_shared": len(shared_ids),
        "shared_genes": [
            {"symbol": first[eid]["symbol"], "ensembl_id": eid}
            for eid in shared_ids
        ],
    }


# ============================================================
# BioNeMo NIM (STUB — wire your endpoint here)
# ============================================================

BIONEMO_BASE = os.environ.get("BIONEMO_BASE_URL", "")


def call_bionemo_nim(model: str, payload: dict) -> dict:
    """Generic NIM call. Example models you might wire:
        - 'meta/esm2nv'           (protein embeddings)
        - 'nvidia/molmim-generate' (molecule generation)
        - 'nvidia/diffdock'        (protein-ligand docking)
        - 'nvidia/alphafold2'      (structure prediction)
    """
    if not BIONEMO_BASE:
        return {"error": "BIONEMO_BASE_URL not set; this is a stub.",
                "model": model, "echo": payload}
    url = f"{BIONEMO_BASE.rstrip('/')}/v1/biology/{model}"
    headers = {}
    if (key := os.environ.get("BIONEMO_API_KEY")):
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()
