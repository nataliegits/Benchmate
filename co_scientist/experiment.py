"""Design an alamarBlue experiment for a hypothesis, and refine a hypothesis
from a bench result. LLM-backed (Claude, via co_scientist.llm).

This is the reasoning at the two ends of the wet-lab loop:
  design_experiment: hypothesis  -> the cleanest alamarBlue test + reagents
  refine_hypothesis: result      -> a sharper hypothesis + the next test

Both are constrained to what the DIY rig can actually do (a colorimetric
viability readout), so the plans are runnable, not aspirational.
"""
from __future__ import annotations

from .llm import call_json

RIG_CAPABILITY = (
    "The ONLY assay available is a colorimetric alamarBlue (resazurin) cell-"
    "viability readout on a small DIY rig: dose cells with a COMPOUND from the "
    "freezer, incubate, add alamarBlue, and read the red/blue ratio over minutes. "
    "It measures metabolic viability (kill vs no-kill) across a few doses plus "
    "controls: not mechanism, localisation, expression, or protein interactions.\n"
    "There is NO capacity for genetic manipulation: no shRNA/siRNA, no CRISPR, no "
    "transfection, no overexpression, no immunoblot, no co-IP, no pulse-chase.\n"
    "So if the hypothesis is genetic (a knockdown, an overexpression, a "
    "protein-level claim), do NOT propose those. Instead design the closest "
    "PHARMACOLOGICAL proxy: pick a small molecule that perturbs the same pathway "
    "arm, and be explicit about what that proxy can and cannot establish."
)


def project_evidence() -> str:
    """Everything the project already knows, as a short prompt block: where the
    hypothesis sits on the leaderboard, what the independent models said about
    it, and any bench results on record.

    This is what makes the design step *informed*: without it the designer only
    sees a sentence of text and can't know that, say, three external models
    disagree with the idea it's about to test.
    """
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    bits = []

    st = repo / "state.json"
    if st.exists():
        try:
            hyps = sorted(json.loads(st.read_text()).get("hypotheses", []),
                          key=lambda h: h.get("elo", 0), reverse=True)[:3]
            if hyps:
                bits.append("Leaderboard (Elo):\n" + "\n".join(
                    f"  {round(h.get('elo', 0))}: {h.get('statement','')[:110]}"
                    for h in hyps))
        except Exception:
            pass

    xc = []
    for name, label in (("variant", "AlphaGenome (regulatory)"),
                        ("boltz", "Boltz (binding)"),
                        ("opentargets", "Open Targets (disease link)"),
                        ("depmap", "DepMap (dependency)"),
                        ("alphamissense", "AlphaMissense (pathogenicity)")):
        f = repo / "benchmark" / f"{name}_scores.json"
        if f.exists():
            try:
                rows = json.loads(f.read_text())
                if rows:
                    best = max(rows, key=lambda r: r.get("score", 0))
                    xc.append(f"  {label}: highest-scoring = {best.get('label')} "
                              f"({best.get('score'):.3g}), n={len(rows)}")
            except Exception:
                pass
    if xc:
        bits.append("Independent cross-check models on record:\n" + "\n".join(xc))

    try:
        from . import assay
        recs = [assay.assay_evidence(l) for l in assay.available_assays()]
        recs = [r for r in recs if r]
        if recs:
            bits.append("Bench results already on record:\n" + "\n".join(
                f"  {r['drug']} on {r['cell_line']} → {r['viability']['verdict']} "
                f"({r['direction_for_benchmate']})" for r in recs))
    except Exception:
        pass

    return "\n\n".join(bits)


def design_experiment(hypothesis: str, evidence: str = "") -> dict:
    """Return a runnable alamarBlue design for `hypothesis` as a dict with keys:
    aim, cell_line, treatment, comparison, controls[], reagents_needed[],
    readout, key_confound, limitation.

    Pass `evidence` (see project_evidence()) so the design accounts for the
    ranking, the cross-check models, and anything already run at the bench.
    """
    ev = (f"WHAT THE PROJECT ALREADY KNOWS. Take this into account. If the "
          f"independent models disagree with this hypothesis, say so plainly in "
          f'"limitation" and design the comparison that would discriminate. '
          f"Never repeat an experiment already on record:\n{evidence}\n\n"
          if evidence.strip() else "")
    return call_json(
        f"Hypothesis to test:\n{hypothesis}\n\n{ev}{RIG_CAPABILITY}\n\n"
        "Design the cleanest alamarBlue experiment to test it.\n"
        "FORMAT RULES: follow exactly:\n"
        "- Every value is a PLAIN STRING (or a list of plain strings). No nested "
        "objects.\n"
        "- Keep each value under 40 words. Be concrete, not exhaustive.\n"
        "- Never invent a compound's mechanism of action. Use only well-established "
        "pharmacology; if no good small-molecule proxy exists for this hypothesis, "
        "say so in \"limitation\" rather than inventing one.\n\n"
        "Output a JSON object:\n"
        '  "aim": one sentence. What this experiment decides,\n'
        '  "cell_line": a real, appropriate human cell line to use,\n'
        '  "treatment": the compound + dose range + timepoint,\n'
        '  "comparison": the key treated-vs-control comparison that answers the aim,\n'
        '  "controls": list of controls, each "name. Why it\'s needed" '
        "(include a vehicle control and a positive control),\n"
        '  "reagents_needed": list of reagent NAMES to pull from the freezer. The '
        "experimental compounds only (assume alamarBlue, media, and plates are on "
        "hand),\n"
        '  "readout": what you measure, and which result would SUPPORT vs REFUTE '
        "the hypothesis,\n"
        '  "key_confound": the single artifact or confound most likely to fool you,\n'
        '  "limitation": what this viability assay CANNOT establish about the '
        "hypothesis: say so plainly, especially if you substituted a "
        "pharmacological proxy for a genetic manipulation.",
        role="generation", max_tokens=3000, temperature=0.4,
    )


def refine_hypothesis(hypothesis: str, result_summary: str) -> dict:
    """Given a bench result, propose how to update the hypothesis. Returns a dict
    with keys: verdict, revised_hypothesis, rationale, next_experiment."""
    return call_json(
        f"Original hypothesis:\n{hypothesis}\n\n"
        f"Bench result from the alamarBlue assay:\n{result_summary}\n\n"
        "Update the thinking in light of the result. Output a JSON object:\n"
        '  "verdict": one of "supported" | "weakened" | "inconclusive",\n'
        '  "revised_hypothesis": a sharper hypothesis that accounts for this result '
        "(if weakened, pivot to the most plausible alternative mechanism or "
        "combination),\n"
        '  "rationale": two sentences on why,\n'
        '  "next_experiment": the single most informative next test (runnable on '
        "the same viability rig where possible).",
        role="generation", max_tokens=800, temperature=0.5,
    )
