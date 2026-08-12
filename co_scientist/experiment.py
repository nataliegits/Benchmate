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



# ---------------------------------------------------------------------------
# What you can actually measure
# ---------------------------------------------------------------------------
# Benchmate started around one assay because that's the rig that exists on my
# kitchen table. But the loop is not about alamarBlue, it's about closing the
# gap between a hypothesis and a number. Any readout that lands as a CSV works
# the same way, so the assay is a parameter rather than an assumption.

ASSAYS: dict[str, dict] = {
    "viability": {
        "label": "Viability (alamarBlue / resazurin)",
        "question": "Are the cells alive after treatment?",
        "answers": "kill versus no-kill across a dose range",
        "cannot": ("mechanism, localisation, expression, or protein "
                   "interactions. It also cannot separate a compound that "
                   "kills from one that merely slows metabolism"),
        "readout": "red/blue ratio over time, then viability as % of control",
        "csv": "t_s, R, G, B, red_blue (from the rig), or well/value",
        "rig": ("Runs on a $60 DIY reader: a colour sensor and a "
                "microcontroller. Any plate reader works too."),
    },
    "qpcr": {
        "label": "Gene expression (qPCR)",
        "question": "Did the transcript level actually change?",
        "answers": "fold change in a target gene versus a reference gene",
        "cannot": ("protein level, localisation, or whether the change "
                   "matters for survival. Transcript is not protein"),
        "readout": "delta delta Ct, expressed as fold change versus control",
        "csv": "sample, target, Ct (one row per well), plus a reference gene",
        "rig": ("Needs a thermocycler, so this is a core-facility or shared "
                "instrument experiment rather than a benchtop one."),
    },
    "proliferation": {
        "label": "Proliferation (growth over time)",
        "question": "Did the cells keep dividing?",
        "answers": "growth rate and doubling time versus control",
        "cannot": ("whether slowed growth is arrest or death, and nothing "
                   "about mechanism. Pair it with viability to tell those "
                   "apart"),
        "readout": "cell count or confluence at intervals, then growth rate",
        "csv": "time_h, condition, value (count or % confluence)",
        "rig": ("Cell counts, confluence from an imager, or the same "
                "colorimetric rig read at intervals."),
    },
}


def assay_capability(assay: str = "viability") -> str:
    """The capability block for the chosen assay, in the same shape as the
    original rig constraint: what it can measure, and what it cannot."""
    a = ASSAYS.get(assay, ASSAYS["viability"])
    return (
        f"The assay available is: {a['label']}.\n"
        f"It answers: {a['question']} Specifically {a['answers']}.\n"
        f"The readout is {a['readout']}, exported as a CSV ({a['csv']}).\n"
        f"It CANNOT establish {a['cannot']}.\n"
        "There is NO capacity for genetic manipulation: no shRNA/siRNA, no "
        "CRISPR, no transfection, no overexpression, no immunoblot, no co-IP, "
        "no pulse-chase.\n"
        "So if the hypothesis is genetic (a knockdown, an overexpression, a "
        "protein-level claim), do NOT propose those. Design the closest "
        "PHARMACOLOGICAL proxy: a small molecule that perturbs the same "
        "pathway arm, and be explicit about what that proxy cannot establish."
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

        # Which genes is this hypothesis about? Used as the fallback match.
        want_genes: set[str] = set()
        if hypothesis:
            try:
                from . import hypothesis_scan as _hsc
                want_genes = {g.upper()
                              for g in _hsc.genes_in(hypothesis,
                                                     validate=False)[0]}
            except Exception:
                pass

        for model in ("Open Targets", "DepMap", "AlphaMissense",
                      "AlphaGenome", "Boltz"):
            for r in _ls.load(model):
                target = str(r.get("target") or "")
                how = ""
                if not hypothesis:
                    how = ""
                elif _same_hypothesis(r.get("label", ""), hypothesis):
                    how = ""
                elif want_genes and target.split()[0].upper() in want_genes:
                    # Cross-check and Design can be looking at differently
                    # worded statements about the same gene: the panel records
                    # against the text you scored, the designer asks about the
                    # text you're designing for. A gene-level score is still
                    # the relevant evidence, so keep it and say where it came
                    # from rather than silently dropping it.
                    how = " (scored on a related hypothesis about this gene)"
                else:
                    continue
                live.append(f"  {model} on {target or r.get('label')}: "
                            f"{r.get('score')}{how}")
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


def design_experiment(hypothesis: str, evidence: str = "",
                      assay: str = "viability") -> dict:
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
        f"Hypothesis to test:\n{hypothesis}\n\n{ev}{assay_capability(assay)}\n\n"
        f"Design the cleanest {ASSAYS.get(assay, ASSAYS['viability'])['label']} "
        f"experiment to test it.\n"
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
            "Hold the verdict to that limitation. An assay that could never have "
            "established the mechanism has not weakened it.\n\n"
            "Most hypotheses here have two parts: a prediction this assay CAN "
            "test, usually a viability or dose-response claim, and a mechanistic "
            "claim it cannot. Score those separately.\n"
            "  * the testable prediction was met, mechanism out of reach\n"
            "      -> \"supported in part\"\n"
            "  * the testable prediction failed\n"
            "      -> \"weakened\"\n"
            "  * everything the assay could reach passed\n"
            "      -> \"supported\"\n"
            "  * the assay could not reach any part of it, or the data was not "
            "usable\n"
            "      -> \"inconclusive\"\n\n"
            "Reserve \"inconclusive\" for a run that taught nothing. If a "
            "pre-specified numeric criterion was met, the result is at least "
            "\"supported in part\" no matter how much mechanism is still open. "
            "Calling a met criterion inconclusive understates the run and is "
            "the more misleading error of the two.\n\n")
    return call_json(
        f"Original hypothesis:\n{hypothesis}\n\n"
        f"{ctx}"
        f"Bench result from the alamarBlue assay:\n{result_summary}\n\n"
        # The prompt used to say "runnable on the same viability rig" without
        # ever saying what the rig is, so the model proposed plate-reader
        # protocols: a single endpoint after a 4 h incubation, for instance.
        # This rig logs the red/blue ratio continuously, and the whole reason
        # viability is computed as a delta is that each well starts at its own
        # offset. A single endpoint throws away the kinetics and puts that
        # offset back into the number.
        f"WHAT THE INSTRUMENT IS:\n{RIG_CAPABILITY}\n\n"
        "The rig logs continuously, so any next experiment must keep the "
        "kinetic read. Do NOT propose a single endpoint reading, and do not "
        "propose dropping the early timepoints: viability is computed as "
        "plateau minus baseline, and without the baseline window there is no "
        "per-well correction. A flagged artifact is a reason to re-read the "
        "well, not a reason to change the readout.\n\n"
        "If the result you were given covers only one condition, or has no "
        "vehicle control to normalise against, say that FIRST and in plain "
        "words, because it is the one thing the user can act on immediately. "
        "Do not bury it inside a mechanistic argument.\n\n"
        "If the next experiment is the same one that was already designed, say "
        "so explicitly and say what to change about how it is run, rather than "
        "restating the design as though it were new.\n\n"
        # Lead with the result. An earlier version of this prompt produced
        # rationales that opened on what the assay could not do, which reads as
        # a failed run even when a pre-specified criterion was met.
        "Say what was learned BEFORE what was not. Open the rationale with the "
        "number the experiment produced and whether it met the criterion. Put "
        "the limits second, in one sentence. Never open on a limitation.\n\n"
        "Update the thinking in light of the result. Output a JSON object:\n"
        '  "verdict": one of "supported" | "supported in part" | "weakened" '
        '| "inconclusive",\n'
        '  "revised_hypothesis": a sharper hypothesis that accounts for this '
        "result. Keep what the data supported, stated with its number, and "
        "narrow what is still open. If weakened, pivot to the most plausible "
        "alternative mechanism or combination,\n"
        '  "rationale": two sentences. The first states the result and whether '
        "it met the criterion. The second states what remains untested,\n"
        '  "next_experiment": the single most informative next test, runnable on '
        "the rig described above.",
        role="generation", max_tokens=800, temperature=0.5,
    )


# ---------------------------------------------------------------------------
# The bench protocol
# ---------------------------------------------------------------------------

def protocol_for(design: dict, plate: str = "96-well",
                 assay: str = "viability") -> dict:
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

    if assay == "qpcr":
        return _protocol_qpcr(design, cell, reagents, controls, treat)
    if assay == "proliferation":
        return _protocol_proliferation(design, cell, reagents, controls, treat,
                                       plate)

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


def protocol_text(design: dict, assay: str = "viability") -> str:
    """The protocol as plain text, for a lab notebook or a printout."""
    p = protocol_for(design, assay=assay)
    out = [ASSAYS.get(assay, ASSAYS["viability"])["label"], "",
           "MATERIALS"]
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


def _protocol_qpcr(design, cell, reagents, controls, treat) -> dict:
    """Standard delta delta Ct workflow. The chemistry-specific numbers stay
    generic on purpose: cycling conditions belong to your kit's insert, and a
    model inventing an annealing temperature is how you get a failed plate."""
    materials = [
        f"{cell} cells, treated as in the design",
        "RNA extraction kit, or TRIzol",
        "DNase I, to remove genomic DNA carryover",
        "Reverse transcription kit",
        "qPCR master mix (SYBR or probe based)",
        "Primers for the target gene, and for a reference gene",
        "96-well qPCR plate and optical seals",
    ] + [f"{r} (from the freezer)" for r in reagents]
    steps = [
        f"Treat {cell}: {treat}. Include the controls in the same experiment, "
        f"harvested at the same time:",
    ] + [f"    - {c}" for c in controls] + [
        "Harvest and extract RNA. Work fast and cold; RNA degrades while you "
        "decide what to do next.",
        "Check RNA quality and concentration. A 260/280 near 2.0 is what you "
        "want. Do not skip this and then wonder why the Ct values scatter.",
        "DNase-treat the RNA. Without it, genomic DNA amplifies and inflates "
        "your apparent expression.",
        "Reverse transcribe an equal mass of RNA per sample. Equal mass is the "
        "whole basis of the comparison.",
        "Include a no-reverse-transcriptase control for at least one sample. "
        "If it amplifies, you are measuring DNA, not transcript.",
        "Set up the qPCR plate in technical triplicate for both the target and "
        "the reference gene, plus a no-template control per primer pair.",
        "Run the cycling program from your master mix's insert, and finish "
        "with a melt curve if you are using SYBR.",
        "Check the melt curve is a single peak per primer pair. Two peaks "
        "means you are amplifying more than one thing.",
        "Export the Ct table as CSV (sample, target, Ct) and drop it into "
        "Benchmate's Results tab, which computes delta delta Ct against your "
        "control and reference gene.",
    ]
    timing = [
        "Treatment: the window from the design",
        "Harvest to cDNA: about 3 h",
        "qPCR run: 1.5 to 2 h",
        "Total: one full day, comfortably",
    ]
    notes = [
        "Transcript is not protein. A fold change tells you the message "
        "level moved, not that the protein or the phenotype followed.",
        "Your reference gene must not respond to the treatment. Under proteasome "
        "inhibition or ER stress, common housekeepers can shift, which silently "
        "distorts every fold change on the plate.",
        "Technical triplicates measure your pipetting. Biological replicates "
        "measure the biology. You need both, and only the second kind counts "
        "as an n.",
    ]
    return _finish(design, materials, steps, timing, notes)


def _protocol_proliferation(design, cell, reagents, controls, treat, plate) -> dict:
    """Growth over time. The measurement is easy; the seeding density and the
    confluence ceiling are what actually decide whether the result means
    anything."""
    materials = [
        f"{cell} cells, in log-phase growth",
        f"{plate} flat-bottom plate, or several for destructive timepoints",
        "Complete growth medium, pre-warmed",
        "A counting method: haemocytometer, automated counter, or imager",
    ] + [f"{r} (from the freezer)" for r in reagents]
    steps = [
        f"Seed {cell} at a density low enough that untreated wells are still "
        f"sub-confluent at your last timepoint. If the control saturates, every "
        f"treated condition looks better than it is.",
        "Let the cells attach overnight.",
        "Record a time-zero reading before dosing. Without it you cannot tell "
        "a growth difference from a seeding difference.",
        f"Dose the plate: {treat}. Set up the controls alongside:",
    ] + [f"    - {c}" for c in controls] + [
        "Read at regular intervals across at least two doublings. Three "
        "timepoints is a line; two is a guess.",
        "Keep the plate's time out of the incubator short and consistent "
        "between reads. Temperature swings show up as growth artifacts.",
        "Export as CSV (time_h, condition, value) and drop it into Benchmate's "
        "Results tab, which fits the growth rate and compares doubling times.",
    ]
    timing = [
        "Seeding to dosing: overnight",
        "Reading window: 48 to 96 h, depending on doubling time",
        "Hands-on: about 15 minutes per timepoint",
    ]
    notes = [
        "Slower growth is not death. A cytostatic compound and a cytotoxic one "
        "produce the same flat curve, which is why this pairs naturally with a "
        "viability readout.",
        "Confluence is not proportional to cell number once cells start "
        "crowding. Stay in the linear range or the effect size is meaningless.",
        "Compare doubling times, not endpoint values. An endpoint difference "
        "can come entirely from an uneven seed.",
    ]
    return _finish(design, materials, steps, timing, notes)


def _finish(design, materials, steps, timing, notes) -> dict:
    """Attach the design's own limitation and confound to any protocol."""
    lim = str((design or {}).get("limitation") or "")
    conf = str((design or {}).get("key_confound") or "")
    if lim:
        notes.append(f"From the design, what this cannot establish: {lim}")
    if conf:
        notes.append(f"Watch for: {conf}")
    return {"materials": materials, "steps": steps, "timing": timing,
            "notes": notes}
