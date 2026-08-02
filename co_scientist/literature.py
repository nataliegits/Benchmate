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


def search(query: str, max_results: int = 8) -> list[dict]:
    """PubMed search, as plain dicts the UI can render.

    Reuses the same E-utilities client the agent uses, so what you see here is
    what it would see.
    """
    from .tools import pubmed_search
    out = []
    for p in pubmed_search(query, max_results=max_results):
        out.append({"pmid": p.pmid, "title": p.title,
                    "abstract": (p.abstract or "")[:1200],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/"})
    return out


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
    for p in search(q, max_results=3):
        print(f"\n{p['pmid']}  {p['title'][:90]}")
        print(f"  {p['abstract'][:160]}...")
