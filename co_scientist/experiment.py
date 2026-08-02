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


def _same_hypothesis(a: str, b: str) -> bool:
    """Loose match between a recorded label and a hypothesis statement.

    Labels are truncated when stored, so compare normalised prefixes rather
    than requiring equality.
    """
    import re as _re
    na = _re.sub(r"\s+", " ", str(a or "")).strip().lower()
    nb = _re.sub(r"\s+", " ", str(b or "")).strip().lower()
    if not na or not nb:
        return False
    return na == nb or na.startswith(nb[:60]) or nb.startswith(na[:60])


def project_evidence(hypothesis: str = "") -> str:
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

    # Cross-check scores for THIS hypothesis, from the live panel.
    #
    # This used to read benchmark/{model}_scores.json and report the
    # highest-scoring entry, but those are gold-set benchmark files: the
    # designer was being told "the best gene in an unrelated benchmark was
    # SYVN1", which is noise at best and misleading at worst. What matters is
    # what the models said about the hypothesis actually being tested.
    live = []
    try:
        from benchmark import live_scores as _ls
        for model in ("Open Targets", "DepMap", "AlphaMissense",
                      "AlphaGenome", "Boltz"):
            for r in _ls.load(model):
                if hypothesis and not _same_hypothesis(r.get("label", ""),
                                                       hypothesis):
                    continue
                live.append(f"  {model} on {r.get('target') or r.get('label')}: "
                            f"{r.get('score')}")
    except Exception:
        pass
    if live:
        bits.append(
            "Independent models scored THIS hypothesis:\n" + "\n".join(live)
            + "\nIf these disagree with the ranking, say so in \"limitation\" "
              "and design the comparison that would discriminate.")
    elif hypothesis:
        bits.append("No independent model has scored this hypothesis yet, so "
                    "the design cannot lean on external support.")

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


def refine_hypothesis(hypothesis: str, result_summary: str,
                      design: dict | None = None) -> dict:
    """Given a bench result, propose how to update the hypothesis. Returns a dict
    with keys: verdict, revised_hypothesis, rationale, next_experiment.

    `design` is the experiment that produced the result. Without it the model
    was interpreting a number in a vacuum: it could call a hypothesis refuted
    when the design's own stated limitation says this assay could never have
    established that in the first place. Passing the design makes the verdict
    accountable to what was actually run.
    """
    ctx = ""
    if design:
        def _s(k):
            v = design.get(k)
            if isinstance(v, (list, tuple)):
                return "; ".join(str(x) for x in v)
            return str(v or "")
        ctx = (
            "The experiment that produced this result:\n"
            f"  Aim: {_s('aim')}\n"
            f"  Cell line: {_s('cell_line')}\n"
            f"  Treatment: {_s('treatment')}\n"
            f"  Comparison: {_s('comparison')}\n"
            f"  Controls: {_s('controls')}\n"
            f"  Readout: {_s('readout')}\n"
            f"  Known confound: {_s('key_confound')}\n"
            f"  STATED LIMITATION: {_s('limitation')}\n\n"
            "Hold the verdict to that limitation. If the assay could not have "
            "established the mechanism either way, the honest verdict is "
            "\"inconclusive\", not \"weakened\". Say plainly which part of the "
            "hypothesis this result can and cannot speak to.\n\n")
    return call_json(
        f"Original hypothesis:\n{hypothesis}\n\n"
        f"{ctx}"
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


# ---------------------------------------------------------------------------
# The bench protocol
# ---------------------------------------------------------------------------

def protocol_for(design: dict, plate: str = "96-well") -> dict:
    """A step-by-step alamarBlue protocol for a given design.

    Built from a fixed template with the design's own values slotted in, not
    generated freely. alamarBlue is a standard assay with a standard procedure,
    and a language model improvising incubation times or dye concentrations is
    a way to get confidently wrong numbers into a lab notebook. The parts that
    vary by experiment (cell line, compound, dose range, controls) come from
    the design; everything else is the published method.

    Returns {materials, steps, timing, notes} with steps as plain strings.
    """
    def _s(k, default=""):
        v = (design or {}).get(k)
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return str(v or default)

    cell = _s("cell_line", "your cell line")
    treat = _s("treatment", "the compound, across a dose range")
    controls = _s("controls") or ["Vehicle control (DMSO, matched to the "
                                  "highest compound volume)",
                                  "Positive kill control"]
    if isinstance(controls, str):
        controls = [controls]
    reagents = _s("reagents_needed") or []
    if isinstance(reagents, str):
        reagents = [reagents]

    materials = [
        f"{cell} cells, in log-phase growth",
        f"{plate} flat-bottom tissue-culture plate",
        "alamarBlue (resazurin) reagent, 10x",
        "Complete growth medium, pre-warmed",
        "DMSO for vehicle control",
    ] + [f"{r} (from the freezer)" for r in reagents]

    steps = [
        f"Seed {cell} into the {plate} plate at your standard density in "
        f"100 uL medium per well. Leave the outer wells medium-only: they "
        f"evaporate fastest and will skew the edges.",

        "Include at least three wells of medium with no cells. This is the "
        "blank, and every later reading gets it subtracted.",

        "Let the cells attach and recover overnight in the incubator.",

        f"Dose the plate: {treat}. Run each condition in triplicate; a single "
        f"well cannot distinguish a real effect from a pipetting error.",

        "Set up the controls in the same plate, not a separate one:",
    ] + [f"    - {c}" for c in controls] + [

        "Return the plate to the incubator for the treatment window in the "
        "design.",

        "Add alamarBlue to 10% of the well volume (10 uL into 100 uL). Add it "
        "to the blank wells too.",

        "Return to the incubator, protected from light. Resazurin is "
        "light-sensitive and a bright bench will cost you signal.",

        "Read the colour: blue means the cells are not reducing the dye, pink "
        "means they are. On the DIY rig, start logging before the colour "
        "shifts so the baseline is captured.",

        "Read again at intervals across the incubation. The rate of change is "
        "more informative than any single endpoint, and it tells you whether "
        "the signal has plateaued.",

        "Export the run as CSV and drop it into Benchmate's Results tab. It "
        "subtracts the blank, audits for artifacts, and computes viability "
        "against the vehicle control.",
    ]

    timing = [
        "Seeding to dosing: overnight (cells must be attached and cycling)",
        "Dosing to dye: the treatment window from the design",
        "Dye to first read: 1 to 4 h is typical; longer for slow-growing or "
        "low-density cultures",
        "Total hands-on time: about 45 minutes across two days",
    ]

    notes = [
        "alamarBlue measures metabolic reduction, not cell number. A compound "
        "that slows metabolism without killing looks identical to one that "
        "kills, so treat this as viability, not cytotoxicity.",
        "The dye is not inert over long incubations. Reading the same wells "
        "for many hours is fine for a rate; comparing across plates read at "
        "different times is not.",
        "Never compare a treated well to an untreated well on a different "
        "plate. Vehicle control belongs in the same plate, on the same day.",
    ]
    lim = _s("limitation")
    if lim:
        notes.append(f"From the design, what this cannot establish: {lim}")
    conf = _s("key_confound")
    if conf:
        notes.append(f"Watch for: {conf}")

    return {"materials": materials, "steps": steps, "timing": timing,
            "notes": notes}


def protocol_text(design: dict) -> str:
    """The protocol as plain text, for a lab notebook or a printout."""
    p = protocol_for(design)
    out = ["alamarBlue viability assay", "", "MATERIALS"]
    out += [f"  - {m}" for m in p["materials"]]
    out += ["", "PROCEDURE"]
    n = 0
    for s in p["steps"]:
        if s.startswith("    "):
            out.append(f"      {s.strip()}")
        else:
            n += 1
            out.append(f"  {n}. {s}")
    out += ["", "TIMING"] + [f"  - {t}" for t in p["timing"]]
    out += ["", "NOTES"] + [f"  - {x}" for x in p["notes"]]
    return "\n".join(out)
