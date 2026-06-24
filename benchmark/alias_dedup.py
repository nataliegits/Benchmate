"""Alias / duplicate detection — where the ontology beats text similarity.

Two hypotheses can be the SAME idea worded differently: "multiple myeloma" vs
"plasma cell myeloma", "ERAD" vs "ER-associated protein degradation". A text
embedding sees different tokens; ontology similarity resolves both to the same
canonical entities and catches the duplicate. This is a win the LLM's parametric
knowledge can't hand you for free — it's about canonical IDENTITY, not recall —
which is exactly why it dodges the "the model already knows it" ceiling that
flattened the discrimination test.

The metric is SEPARATION: how much higher does a method score genuine duplicates
("same") than unrelated pairs ("different")? Bigger gap = better at telling them
apart.

Two honest caveats baked into the data:
  • OntoMCP unifies disease/process aliases well, but currently does NOT unify
    human gene symbols (HRD1 vs SYVN1) — the coverage gap. That pair is included
    and labelled, and it's expected to FAIL until HGNC/mygene normalisation is
    added. It's the honest asterisk on this result.
  • The embedding baseline is the repo's PLACEHOLDER hash embedding
    (co_scientist.tools.embed). A real sentence-embedder would score higher, but
    it still wouldn't know HRD1 = SYVN1 unless trained to. Swap one in to make
    the comparison fully fair.

    python -m benchmark.alias_dedup          # live (OntoMCP for the ontology side)
    python -m benchmark.alias_dedup --demo   # synthetic numbers, no server
"""
from __future__ import annotations

import argparse
import statistics

# label: "same" = the pair is one idea in two guises; "different" = unrelated.
PAIRS: list[dict] = [
    dict(label="same", kind="disease alias",
         a="Bortezomib resistance in multiple myeloma depends on ERAD.",
         b="ERAD drives drug resistance in plasma cell myeloma.",
         note="multiple myeloma = plasma cell myeloma → same MONDO/NCIT term"),
    dict(label="same", kind="process alias",
         a="Blocking ERAD via the HRD1 complex restores proteotoxicity.",
         b="Inhibiting ER-associated protein degradation through the Hrd1p "
           "ligase complex re-imposes proteotoxic stress.",
         note="ERAD = ER-associated protein degradation; HRD1 complex shared"),
    dict(label="same", kind="GENE alias (known gap)",
         a="HRD1 disposes of misfolded light chains in resistant cells.",
         b="SYVN1 clears misfolded immunoglobulin light chains in resistant cells.",
         note="HRD1 = SYVN1 — but OntoMCP doesn't unify them yet (coverage gap)"),
    dict(label="different", kind="distinct mechanism",
         a="HRD1/ERAD dependence drives bortezomib resistance.",
         b="Bone-marrow stromal contact protects myeloma cells via IL-6 signalling.",
         note="genuinely different mechanisms — should score LOW"),
    dict(label="different", kind="distinct mechanism",
         a="EDEM1 mannosidase triage offloads the proteasome.",
         b="UPR signalling status modulates apoptosis in plasma cells.",
         note="different processes — should score LOW"),
]


def summarize(rows: list[dict]) -> dict:
    """rows: [{label, onto, emb}]. Separation = mean(same) - mean(different)
    for each method. Pure + testable."""
    def sep(key):
        same = [r[key] for r in rows if r["label"] == "same"]
        diff = [r[key] for r in rows if r["label"] == "different"]
        if not same or not diff:
            return None
        return statistics.mean(same) - statistics.mean(diff)
    return {"ontology_separation": sep("onto"), "embedding_separation": sep("emb")}


def _verdict(res: dict) -> str:
    o, e = res["ontology_separation"], res["embedding_separation"]
    if o is None or e is None:
        return "need at least one 'same' and one 'different' pair."
    if o > e:
        return ("Ontology separates duplicates from distinct pairs better than the "
                "text embedding — canonical identity beats token overlap.")
    return ("Embedding matched or beat ontology here — check coverage (are the "
            "alias entities resolving?) or that a real embedder isn't masking it.")


def _build_rows_live() -> list[dict]:
    from co_scientist.ontology import ontology_similarity
    from co_scientist.tools import embed, cosine
    rows = []
    for p in PAIRS:
        onto = ontology_similarity(p["a"], p["b"])
        rows.append({**p,
                     "onto": onto if onto is not None else 0.0,
                     "onto_na": onto is None,
                     "emb": cosine(embed(p["a"]), embed(p["b"]))})
    return rows


def _build_rows_demo() -> list[dict]:
    # Illustrative numbers: ontology unifies the disease/process aliases (high on
    # the 'same' pairs) but not the gene alias; embedding is near-random on all.
    demo = [0.67, 0.50, 0.0, 0.0, 0.0]          # ontology Jaccard per PAIR
    emb = [0.31, 0.28, 0.22, 0.26, 0.24]        # placeholder embedding cosine
    return [{**p, "onto": demo[i], "onto_na": demo[i] == 0.0 and p["label"] == "same",
             "emb": emb[i]} for i, p in enumerate(PAIRS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic numbers, no OntoMCP")
    args = ap.parse_args()

    rows = _build_rows_demo() if args.demo else _build_rows_live()
    res = summarize(rows)

    print("=" * 72)
    print("ALIAS / DUPLICATE DETECTION  —  ontology similarity vs text embedding")
    print("=" * 72)
    print(f"{'label':10} {'ontology':>9} {'embed':>7}   pair")
    for r in rows:
        flag = "  ← gap: not unified" if r.get("onto_na") else ""
        print(f"{r['label']:10} {r['onto']:>9.2f} {r['emb']:>7.2f}   {r['kind']}{flag}")
    print("-" * 72)
    o, e = res["ontology_separation"], res["embedding_separation"]
    print(f"  separation (same − different)   ontology {o:+.2f}   embedding {e:+.2f}")
    print(f"  {_verdict(res)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
