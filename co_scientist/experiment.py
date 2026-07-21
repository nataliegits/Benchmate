"""Design an alamarBlue experiment for a hypothesis, and refine a hypothesis
from a bench result. LLM-backed (Claude, via co_scientist.llm).

This is the reasoning at the two ends of the wet-lab loop:
  design_experiment  — hypothesis  -> the cleanest alamarBlue test + reagents
  refine_hypothesis  — result      -> a sharper hypothesis + the next test

Both are constrained to what the DIY rig can actually do (a colorimetric
viability readout), so the plans are runnable, not aspirational.
"""
from __future__ import annotations

from .llm import call_json

RIG_CAPABILITY = (
    "The only assay available is a colorimetric alamarBlue (resazurin) cell-"
    "viability readout on a small DIY rig: dose cells with a compound, incubate, "
    "add alamarBlue, and read the red/blue ratio over minutes. It measures "
    "metabolic viability (kill vs no-kill) across a few doses plus controls — "
    "not mechanism, localisation, or expression."
)


def design_experiment(hypothesis: str) -> dict:
    """Return a runnable alamarBlue design for `hypothesis` as a dict with keys:
    aim, cell_line, treatment, comparison, controls[], reagents_needed[],
    readout, key_confound."""
    return call_json(
        f"Hypothesis to test:\n{hypothesis}\n\n{RIG_CAPABILITY}\n\n"
        "Design the cleanest alamarBlue experiment to test it. Output a JSON object:\n"
        '  "aim": one sentence — what this experiment decides,\n'
        '  "cell_line": a real, appropriate human cell line to use,\n'
        '  "treatment": the compound + dose range + timepoint,\n'
        '  "comparison": the key treated-vs-control comparison that answers the aim,\n'
        '  "controls": list of controls, each "name — why it\'s needed" '
        "(include a vehicle control and a positive control),\n"
        '  "reagents_needed": list of reagent NAMES to pull from the freezer — the '
        "experimental compounds only (assume alamarBlue, media, and plates are on "
        "hand),\n"
        '  "readout": what you measure, and which result would SUPPORT vs REFUTE '
        "the hypothesis,\n"
        '  "key_confound": the single artifact or confound most likely to fool you.',
        role="generation", max_tokens=1100, temperature=0.4,
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
