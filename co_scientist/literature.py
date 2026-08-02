"""Papers you pinned, handed to the generation agent alongside its own search.

The agent already searches PubMed. It writes its own seed queries, pulls three
abstracts each, and reasons over them. All of that happens inside a subprocess
where you never see it, so the literature half of the evidence is invisible
while the Geneformer half sits in a tab.

This closes that gap from the other direction. You search PubMed yourself, pin
the papers you think matter, and they go into the same prompt the agent builds.
Your pinned set does not replace the agent's search, it is added to it, because
the point of the agent searching is to find things you would not have thought
to look for.

Pinned papers live in data/literature/pinned.json so a run picks them up the
same way it picks up the Geneformer cache.
"""
from __future__ import annotations

import json
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "data" / "literature"
PINNED = DIR / "pinned.json"
MAX_IN_PROMPT = 8          # abstracts are long and context is not free


# Words that carry no search value. PubMed ANDs every term it is given, so a
# question typed in full asks it for papers containing the word "what", and
# comes back empty. Stripping these is the difference between zero results and
# a useful set.
_STOP = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "is", "are", "was", "were", "be", "been", "do", "does", "did", "can",
    "could", "would", "should", "will", "what", "which", "who", "whom",
    "whose", "why", "how", "when", "where", "that", "this", "these", "those",
    "there", "their", "its", "it", "we", "i", "you", "my", "our", "your",
    "some", "any", "all", "more", "most", "other", "such", "than", "then",
    "so", "if", "but", "not", "no", "into", "from", "by", "at", "as",
    "drive", "drives", "driving", "cause", "causes", "make", "makes",
    "use", "using", "used", "find", "show", "shows", "study", "studies",
    "role", "effect", "effects", "about", "between", "during", "via",
    "novel", "new", "potential", "possible", "help", "helps",
    # Generic biology nouns. They match almost every paper, so they narrow a
    # search without making it more relevant.
    "gene", "genes", "cell", "cells", "pathway", "pathways", "protein",
    "proteins", "expression", "level", "levels", "human", "line", "lines",
    "activity", "function", "mechanism", "response", "treatment", "target",
    "targets", "related", "associated", "involved",
}


def to_terms(text: str, limit: int = 6) -> list[str]:
    """Search terms from a question, most specific first.

    Gene symbols and other all-caps tokens lead, because they are the terms
    that narrow a PubMed search usefully. Everything else follows in the order
    it was written.
    """
    import re
    # Split on hyphens as well as spaces. "ERAD-pathway" as a single token is
    # not a PubMed term, and lower-casing it loses the gene-like signal in
    # "ERAD".
    raw = re.findall(r"[A-Za-z][A-Za-z0-9]*", str(text or ""))
    caps, words, seen = [], [], set()
    for w in raw:
        low = w.lower()
        if low in _STOP or len(low) < 3 or low in seen:
            continue
        seen.add(low)
        if w.isupper() and len(w) > 1:
            caps.append(w)
        else:
            words.append(low)
    return (caps + words)[:limit]


def build_query(terms) -> str:
    return " AND ".join(terms)


def search(query: str, max_results: int = 8) -> dict:
    """PubMed search that degrades until it finds something.

    Returns {papers, query, tried}. `query` is the search that produced the
    results, which matters: a search that quietly rewrote what you typed and
    said nothing about it would be worse than one that found nothing.

    A plain keyword query is passed through untouched. A question gets reduced
    to terms, then broadened one term at a time until PubMed answers.
    """
    from .tools import pubmed_search

    q = str(query or "").strip()
    attempts: list[str] = []
    if q and "?" not in q and len(q.split()) <= 6:
        attempts.append(q)          # already looks like a search
    terms = to_terms(q)
    for n in range(min(len(terms), 4), 0, -1):
        cand = build_query(terms[:n])
        if cand and cand not in attempts:
            attempts.append(cand)

    tried = []
    for cand in attempts:
        tried.append(cand)
        try:
            hits = pubmed_search(cand, max_results=max_results)
        except Exception:
            continue
        if hits:
            return {"papers": [
                        {"pmid": p.pmid, "title": p.title,
                         "abstract": (p.abstract or "")[:1200],
                         "url": f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/"}
                        for p in hits],
                    "query": cand, "tried": tried}
    return {"papers": [], "query": attempts[-1] if attempts else q,
            "tried": tried}


def load_pinned() -> list[dict]:
    if not PINNED.exists():
        return []
    try:
        rows = json.loads(PINNED.read_text())
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def pin(paper: dict) -> None:
    """Add a paper, keyed on PMID so pinning twice is harmless."""
    rows = [r for r in load_pinned() if r.get("pmid") != paper.get("pmid")]
    rows.append(paper)
    DIR.mkdir(parents=True, exist_ok=True)
    PINNED.write_text(json.dumps(rows, indent=2))


def unpin(pmid: str) -> None:
    rows = [r for r in load_pinned() if r.get("pmid") != pmid]
    DIR.mkdir(parents=True, exist_ok=True)
    PINNED.write_text(json.dumps(rows, indent=2))


def clear() -> None:
    if PINNED.exists():
        PINNED.unlink()


def prompt_block() -> str:
    """The pinned papers formatted for the generation prompt.

    Empty string when nothing is pinned, so the caller can concatenate without
    checking.
    """
    rows = load_pinned()[:MAX_IN_PROMPT]
    if not rows:
        return ""
    parts = []
    for r in rows:
        abstract = (r.get("abstract") or "").strip()
        parts.append(f"PMID {r.get('pmid')}: {r.get('title')}\n{abstract}")
    return ("\n\nPapers the user pinned as relevant. Treat these as important "
            "context, and cite the PMID when a hypothesis leans on one:\n"
            + "\n\n".join(parts))


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "SEL1L ERAD bortezomib myeloma"
    r = search(q, max_results=3)
    print(f"query used: {r['query']}   (tried: {r['tried']})")
    for p in r["papers"]:
        print(f"\n{p['pmid']}  {p['title'][:90]}")
        print(f"  {p['abstract'][:160]}...")
