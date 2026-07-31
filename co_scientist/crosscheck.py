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

from functools import lru_cache

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

@lru_cache(maxsize=256)
def gene_missense_burden(gene: str, max_variants: int = 5
                         ) -> tuple[float | None, int, str]:
    """Mean AlphaMissense pathogenicity across a gene's known ClinVar missense
    variants. Returns (mean, n_scored, why_not).

    Every coordinate comes from ClinVar — nothing is generated. Capped at a
    handful of variants because each one is a VEP round-trip and this runs
    inside a button click.

    Cached: the same gene gets asked for on every rerun, and this is ~5 network
    calls deep.
    """
    try:
        from benchmark.fetch_clinvar import fetch
    except Exception as e:
        return None, 0, f"{gene}: ClinVar fetcher unavailable ({e})."

    try:
        recs = fetch([gene], per_class=max_variants)
    except Exception as e:
        return None, 0, f"{gene}: ClinVar lookup failed ({e})."

    path = [r for r in recs
            if str(r.get("clinsig", "")).lower().startswith("patho")]
    if not path:
        return None, 0, (f"{gene}: no pathogenic missense variants on record in "
                         f"ClinVar, so there's no coding-change signal to score.")

    scores = []
    for r in path[:max_variants]:
        s = target_scorer.alphamissense_score(
            r.get("chrom"), r.get("pos"), r.get("ref"), r.get("alt"))
        if s is not None:
            scores.append(s)
    if not scores:
        return None, 0, (f"{gene}: found ClinVar variants but AlphaMissense "
                         f"returned no score for any of them.")
    return sum(scores) / len(scores), len(scores), ""


# back-compat: the UI used to reach for the private name
_gene_missense_burden = gene_missense_burden


# ---------------------------------------------------------------------------
# Show your work
# ---------------------------------------------------------------------------
# A score with no visible provenance is a black box, and a black box is exactly
# what a cross-check is supposed to protect you from. Each explain_* below
# returns the real steps taken — resolved identifiers, endpoints hit, how many
# rows were averaged — so a number can be audited rather than trusted.

def explain_opentargets(symbol: str, disease: str) -> dict:
    """Score `symbol` against `disease`, returning the score AND the lookup."""
    steps: list[str] = []
    ens = did = None
    try:
        ens = target_scorer._resolve(symbol, "target")
        steps.append(f"Resolved gene **{symbol}** → `{ens or 'not found'}` "
                     f"(Open Targets search API)")
        did = target_scorer._resolve(disease, "disease")
        steps.append(f"Resolved disease **{disease}** → `{did or 'not found'}` "
                     f"(EFO / MONDO ontology id)")
    except Exception as e:
        steps.append(f"Identifier lookup failed: {e}")

    score = target_scorer.opentargets_score(symbol, disease)
    steps.append("Asked the GraphQL API for the overall association score "
                 "between those two ids")
    if score is not None:
        steps.append(f"Returned **{score:.3f}** — {read_opentargets(score)}")
    return {"score": score, "steps": steps,
            "endpoint": getattr(target_scorer, "OT_URL", "Open Targets GraphQL"),
            "detail": ("The score aggregates every evidence type Open Targets "
                       "holds for this gene–disease pair: GWAS and rare-disease "
                       "genetics, differential expression, animal models, known "
                       "drugs, pathway membership and text-mined literature. It "
                       "is a weighted harmonic sum, so one strong line of "
                       "evidence lifts it more than several weak ones."),
            "ids": {"target": ens, "disease": did}}


def explain_depmap(symbol: str) -> dict:
    """Dependency score for `symbol`, plus which data it came from."""
    steps: list[str] = []
    src = target_scorer.depmap_source()
    lineage = target_scorer.depmap_lineage_in_use()
    n_lines = None
    raw = None

    if src == "full":
        steps.append(f"Read the full CRISPR knockout matrix "
                     f"(`{target_scorer.DEPMAP_CSV.name}`)")
        try:
            df = target_scorer._depmap_frame()
            col = next((c for c in df.columns
                        if str(c).split(" ")[0].upper() == symbol.upper()), None)
            if col:
                series = df[col]
                ids = target_scorer._myeloma_model_ids()
                if ids:
                    sub = series[series.index.isin(ids)]
                    if len(sub) >= 3:
                        series = sub
                n_lines = int(series.notna().sum())
                raw = float(series.mean())
                steps.append(f"Found column `{col}`")
        except Exception as e:
            steps.append(f"Matrix read failed: {e}")
    elif src == "summary":
        steps.append("Read the precomputed per-gene summary that ships with "
                     "Benchmate (`gene_effect_summary.csv`) — same DepMap "
                     "numbers, condensed so no 440 MB download is needed")
        try:
            row = target_scorer._depmap_summary_frame().loc[symbol.upper()]
            raw = float(row["mean_effect"])
            n_lines = int(row["n_lines"])
        except Exception:
            steps.append(f"{symbol} is not in the summary")

    if n_lines is not None:
        steps.append(f"Averaged the gene-effect score across **{n_lines} "
                     f"{'' if lineage == 'all' else lineage + ' '}cell lines**")
    if raw is not None:
        steps.append(f"Mean gene effect = **{raw:+.3f}** (negative = cells lose "
                     f"fitness without it); reported as **{-raw:.3f}** so higher "
                     f"means more essential")

    score = target_scorer.depmap_score(symbol)
    return {"score": score, "steps": steps, "source": src, "lineage": lineage,
            "n_lines": n_lines, "raw_effect": raw,
            "detail": ("DepMap knocked out this gene with CRISPR in hundreds of "
                       "cancer cell lines and measured how much each line's "
                       "growth suffered (the Chronos gene-effect score). Around "
                       "0 means the cells didn't care; −1 is roughly the median "
                       "of known essential genes. This is measured data, not a "
                       "prediction.")}


def explain_missense(gene: str, max_variants: int = 5) -> dict:
    """Gene-level pathogenicity, showing every variant used to get there."""
    steps: list[str] = []
    variants: list[dict] = []
    try:
        from benchmark.fetch_clinvar import fetch
        recs = fetch([gene], per_class=max_variants)
        steps.append(f"Queried **ClinVar** (NCBI E-utilities) for missense "
                     f"variants in **{gene}**")
        path = [r for r in recs
                if str(r.get("clinsig", "")).lower().startswith("patho")]
        steps.append(f"Kept **{len(path)}** classified pathogenic "
                     f"(of {len(recs)} returned)")
        for r in path[:max_variants]:
            s = target_scorer.alphamissense_score(
                r.get("chrom"), r.get("pos"), r.get("ref"), r.get("alt"))
            variants.append({
                "variant": f"chr{r.get('chrom')}:{r.get('pos')} "
                           f"{r.get('ref')}>{r.get('alt')}",
                "clinvar": r.get("clinsig"), "alphamissense": s})
        scored = [v["alphamissense"] for v in variants
                  if v["alphamissense"] is not None]
        if scored:
            steps.append(f"Scored each one through **Ensembl VEP** with the "
                         f"AlphaMissense plugin, then averaged "
                         f"**{len(scored)}** values")
    except Exception as e:
        steps.append(f"Lookup failed: {e}")

    mean, n, why = gene_missense_burden(gene, max_variants)
    return {"score": mean, "n": n, "why": why, "steps": steps,
            "variants": variants,
            "detail": ("AlphaMissense predicts, for a single amino-acid "
                       "substitution, how likely it is to cause disease. It "
                       "needs one specific base change, so to say something "
                       "about a whole gene we take that gene's variants already "
                       "classified pathogenic in ClinVar and average their "
                       "scores. Every coordinate is real — none are generated. "
                       "A high average means coding changes here tend to be "
                       "damaging, which is evidence the protein matters.")}


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
        # Be specific. This file is gitignored (440 MB), so it's present locally
        # and absent on a hosted deploy — "not downloaded" is confusing if you
        # know you downloaded it.
        _p = target_scorer.DEPMAP_CSV
        skipped.append({"model": "DepMap", "kind": "setup",
                        "why": (f"no CRISPR matrix at `{_p}`. It's a 440 MB file "
                                f"excluded from git, so it has to be downloaded "
                                f"once per machine — and it won't be present on "
                                f"a hosted deploy at all.")})
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

    # ---- AlphaMissense ----------------------------------------------------
    # If the hypothesis names a variant, score that variant. If it only names a
    # gene, we can still get a real gene-level answer: pull that gene's known
    # pathogenic missense variants from ClinVar and score those. That says how
    # damaging coding change in this gene tends to be — a genuine signal, from
    # real coordinates, rather than "not applicable".
    if not variants and genes:
        for g in genes:
            mean, n, why = gene_missense_burden(g)
            if mean is None:
                skipped.append({"model": "AlphaMissense", "why": why})
            else:
                rows.append({"model": "AlphaMissense", "target": f"{g} (gene-level)",
                             "score": round(mean, 3),
                             "reading": (f"{read_alphamissense(mean)} — mean over "
                                         f"{n} known ClinVar missense variants"),
                             "scale": "0–1, >0.564 = likely pathogenic"})
    elif not variants:
        skipped.append({"model": "AlphaMissense",
                        "why": ("no gene or variant named, so there's nothing to "
                                "look up.")})
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
    skipped.append({"model": "AlphaGenome", "kind": "setup",
                    "why": ("needs a free API key and runs in Colab — generate "
                            "the notebook below, then upload its scores.")})
    import os
    if not os.environ.get("BOLTZ_API_KEY"):
        skipped.append({"model": "Boltz", "kind": "setup",
                        "why": "needs a paid API key from api.boltz.bio."})

    # Split "can't run" from "doesn't apply" — lumping them together makes a
    # working panel look broken.
    for s in skipped:
        s.setdefault("kind", "na")
    return {"scan": sc, "rows": rows, "skipped": skipped,
            "setup_needed": [s for s in skipped if s["kind"] == "setup"],
            "not_applicable": [s for s in skipped if s["kind"] == "na"],
            "disease": disease}


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
