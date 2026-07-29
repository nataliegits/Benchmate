"""Score one hypothesis against the independent models.

The benchmark/ code answers a different question: "across a fixed gold set,
does my Elo ranking correlate with these predictors?" That's about whether the
LLM judge can be trusted at all. Useful, but it's not what you want when you're
holding one hypothesis and deciding whether to spend a week of bench time on it.

This module answers the second question. Give it a hypothesis; it finds the
genes, asks each model that applies, and reports what each one said — including,
explicitly, which models don't apply and why. "Not applicable" and "broken" look
identical if you don't say which one it is.

Scales differ per model, so each score comes with a plain-language reading
rather than a bare number.
"""
from __future__ import annotations

from . import hypothesis_scan, target_scorer

DEFAULT_DISEASE = getattr(target_scorer, "DEFAULT_DISEASE", "multiple myeloma")


# ---------------------------------------------------------------------------
# What's set up, and what each one needs
# ---------------------------------------------------------------------------

def model_status() -> list[dict]:
    """One row per model: what it asks, whether it's usable right now, and the
    exact thing needed if it isn't.

    `where` is the honest answer to "do I need a terminal for this?":
      live   — runs in the app over a public API, nothing to install
      file   — needs one data file downloaded from a website
      colab  — needs a free API key and a generated notebook
      key    — needs a paid API key
    """
    depmap_csv = target_scorer.DEPMAP_CSV
    model_csv = target_scorer.DEPMAP_MODEL_CSV
    import os

    rows = [
        {"model": "Open Targets", "asks": "Is this gene linked to the disease?",
         "where": "live", "available": target_scorer.opentargets_available(),
         "setup": "Nothing — public API, no key, no download."},
        {"model": "DepMap", "asks": "Do cancer cells need this gene to survive?",
         "where": "file", "available": target_scorer.depmap_available(),
         "setup": (f"Download CRISPRGeneEffect.csv (~440 MB) from "
                   f"depmap.org/portal/data_page/ into {depmap_csv.parent}/ "
                   f"— a browser download, no terminal needed.")},
        {"model": "AlphaMissense",
         "asks": "Is this coding variant likely pathogenic?",
         "where": "live", "available": target_scorer.alphamissense_available(),
         "setup": "Nothing — free Ensembl VEP API. Needs a variant, not just a gene."},
        {"model": "AlphaGenome",
         "asks": "Does this variant change gene expression?",
         "where": "colab", "available": False,
         "setup": ("Free key at alphagenomedocs.com, then generate the Colab "
                   "notebook below and upload the scores it produces.")},
        {"model": "Boltz", "asks": "Does the drug actually bind the target?",
         "where": "key", "available": bool(os.environ.get("BOLTZ_API_KEY")),
         "setup": "Paid API key from api.boltz.bio (launch credits available)."},
    ]

    # Call out the silent-degradation case: without Model.csv the dependency
    # score is a pan-cancer average, not myeloma-specific. It still returns a
    # number, so nothing looks wrong — which is exactly the problem.
    for r in rows:
        if r["model"] == "DepMap" and r["available"] and not model_csv.exists():
            r["caveat"] = (
                f"Model.csv not found, so the score is averaged over ALL cancer "
                f"cell lines rather than {target_scorer.DEPMAP_LINEAGE} lines. "
                f"Download Model.csv from the same DepMap page for an on-target "
                f"number.")
    return rows


# ---------------------------------------------------------------------------
# Reading the numbers
# ---------------------------------------------------------------------------

def read_opentargets(score: float) -> str:
    if score >= 0.5:
        return "strong disease association on record"
    if score >= 0.1:
        return "some association on record"
    if score > 0.0:
        return "weak association"
    return "no association on record for this disease"


def read_depmap(score: float) -> str:
    # score is -mean(gene effect); higher = more essential
    if score >= 1.0:
        return "strong dependency — cells die without it"
    if score >= 0.5:
        return "moderate dependency"
    if score >= 0.2:
        return "weak dependency"
    return "not a dependency in these lines"


def read_alphamissense(score: float) -> str:
    if score >= 0.564:
        return "likely pathogenic (above AlphaMissense's threshold)"
    if score >= 0.34:
        return "ambiguous"
    return "likely benign"


# ---------------------------------------------------------------------------

def score_hypothesis(text: str, disease: str = DEFAULT_DISEASE,
                     validate: bool = True) -> dict:
    """Run every applicable model against one hypothesis.

    Returns {scan, rows, skipped, disease} where `rows` is one record per
    (model, target) with a raw score and its reading, and `skipped` explains
    each model that didn't run.
    """
    sc = hypothesis_scan.scan(text, validate=validate)
    rows: list[dict] = []
    skipped: list[dict] = []

    genes = sc["genes"]
    variants = sc["scoreable_variants"]

    # ---- Open Targets: gene x disease -------------------------------------
    if not target_scorer.opentargets_available():
        skipped.append({"model": "Open Targets",
                        "why": "httpx not installed — can't reach the API."})
    elif not genes:
        skipped.append({"model": "Open Targets",
                        "why": "no gene named in the hypothesis."})
    else:
        for g in genes:
            v = target_scorer.opentargets_score(g, disease)
            if v is None:
                skipped.append({"model": "Open Targets",
                                "why": f"{g} didn't resolve, or the API errored."})
            else:
                rows.append({"model": "Open Targets", "target": g,
                             "score": round(v, 3), "reading": read_opentargets(v),
                             "scale": "0–1, higher = stronger link"})

    # ---- DepMap: gene dependency -----------------------------------------
    if not target_scorer.depmap_available():
        skipped.append({"model": "DepMap",
                        "why": "CRISPRGeneEffect.csv not downloaded yet."})
    elif not genes:
        skipped.append({"model": "DepMap",
                        "why": "no gene named in the hypothesis."})
    else:
        for g in genes:
            v = target_scorer.depmap_score(g)
            if v is None:
                skipped.append({"model": "DepMap",
                                "why": f"{g} isn't in the CRISPR matrix."})
            else:
                rows.append({"model": "DepMap", "target": g,
                             "score": round(v, 3), "reading": read_depmap(v),
                             "scale": "higher = more essential"})

    # ---- AlphaMissense: needs a real variant ------------------------------
    if not variants:
        skipped.append({"model": "AlphaMissense",
                        "why": ("no variant with genomic coordinates in the "
                                "hypothesis. It scores a specific base change, "
                                "so a gene name alone isn't enough.")})
    else:
        for v in variants:
            s = target_scorer.alphamissense_score(
                v["chrom"], v["pos"], v["ref"], v["alt"])
            if s is None:
                skipped.append({"model": "AlphaMissense",
                                "why": (f"{v['raw']} returned no score — it may "
                                        f"not be a missense change.")})
            else:
                rows.append({"model": "AlphaMissense", "target": v["raw"],
                             "score": round(s, 3),
                             "reading": read_alphamissense(s),
                             "scale": "0–1, >0.564 = likely pathogenic"})

    # ---- the two that need external setup ---------------------------------
    skipped.append({"model": "AlphaGenome",
                    "why": ("runs in Colab — generate the notebook below, then "
                            "upload the scores it produces.")})
    import os
    if not os.environ.get("BOLTZ_API_KEY"):
        skipped.append({"model": "Boltz",
                        "why": "no BOLTZ_API_KEY set (paid API)."})

    return {"scan": sc, "rows": rows, "skipped": skipped, "disease": disease}


def verdict(result: dict) -> str:
    """One line on whether the independent models back the hypothesis.

    Deliberately cautious: agreement here is weak evidence, and disagreement is
    a flag to investigate rather than a refutation.
    """
    rows = result["rows"]
    if not rows:
        return ("No model could score this hypothesis yet — see the reasons "
                "below. That's a setup gap, not a judgement on the idea.")
    ot = [r for r in rows if r["model"] == "Open Targets"]
    dm = [r for r in rows if r["model"] == "DepMap"]
    strong_ot = [r for r in ot if r["score"] >= 0.1]
    strong_dm = [r for r in dm if r["score"] >= 0.5]

    if ot and not strong_ot:
        return ("Open Targets has no meaningful association between these genes "
                "and the disease. Worth asking why the ranking liked this — the "
                "link may be novel, or the hypothesis may be off.")
    if strong_ot and strong_dm:
        return ("Both the disease-association and dependency evidence back this. "
                "Independent support, though neither model has seen your "
                "specific mechanism.")
    if strong_ot:
        return ("Disease association holds up, but these genes aren't strong "
                "dependencies in these cell lines — expect a partial effect at "
                "the bench.")
    return ("Mixed signal across the models. Cheap to investigate now; "
            "expensive to discover after the experiment.")
