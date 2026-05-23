"""External tools the agents can call.

Real:
    - PubMed search (NCBI E-utilities, no key required)

Stubbed:
    - BioNeMo NIM call (uncomment when you have an endpoint)
    - Embedding (uses a cheap hash proxy until you wire a real model)
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

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
