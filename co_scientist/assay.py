"""alamarBlue viability assay → Benchmate evidence.

The bench rig (benchmate-demo) logs a colorimetric time-course: as live cells
reduce resazurin (blue) to resorufin (pink), the TCS34725 sensor's red/blue
ratio climbs and plateaus. The *rate* and *plateau* of that climb are the
viability readout — dead cells don't reduce the dye, so their ratio stays flat
near baseline.

This module turns one run's CSV (columns: t_s, R, G, B, red_blue) into:
  1. quantitative metrics (baseline, plateau, Δ, initial slope, AUC, t½),
  2. a viability call (optionally a % vs a vehicle-control Δ), and
  3. a Benchmate **evidence record** (JSON) written to data/rig/, in the same
     spirit as the Geneformer cache — so the Generation/Reflection agents can
     reason about a real bench result, not just the literature.

CLI:
    python -m co_scientist.assay path/to/alamarblue_run.csv \
        --hypothesis p97_CB5083 --drug "CB-5083 (1 uM, 48h)" \
        --cell "RPMI-8226, bortezomib-resistant" --control-delta 0.24
"""
from __future__ import annotations

import argparse
import csv
import re
import json
from pathlib import Path

from co_scientist import audit as _audit

RIG_DIR = Path(__file__).resolve().parent.parent / "data" / "rig"


def read_run(path: str | Path) -> list[dict]:
    """Parse an alamarBlue run CSV into [{t_s, red_blue, R, G, B}, ...]."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t_s": float(r["t_s"]),
                    "red_blue": float(r["red_blue"]),
                    "R": float(r.get("R", "nan") or "nan"),
                    "G": float(r.get("G", "nan") or "nan"),
                    "B": float(r.get("B", "nan") or "nan"),
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        raise ValueError("No usable rows (need columns t_s and red_blue).")
    return rows


def metrics(rows: list[dict], plateau_window_s: float = 60.0) -> dict:
    """Viability metrics from the red/blue kinetic trace."""
    t = [r["t_s"] for r in rows]
    rb = [r["red_blue"] for r in rows]
    duration = t[-1] - t[0]
    baseline = rb[0]

    # plateau = mean over the final `plateau_window_s` of the run
    tail = [v for ti, v in zip(t, rb) if ti >= t[-1] - plateau_window_s]
    plateau = sum(tail) / len(tail)
    delta = plateau - baseline

    # initial slope: change over the first minute, expressed per minute
    first_min = [(ti, v) for ti, v in zip(t, rb) if ti <= t[0] + 60.0]
    if len(first_min) >= 2:
        (t0, v0), (t1, v1) = first_min[0], first_min[-1]
        slope_per_min = (v1 - v0) / max((t1 - t0) / 60.0, 1e-9)
    else:
        slope_per_min = 0.0

    # area under the (ratio - baseline) curve, trapezoidal, in ratio*s
    auc = 0.0
    for a, b in zip(rows, rows[1:]):
        auc += ((a["red_blue"] - baseline) + (b["red_blue"] - baseline)) / 2 * (b["t_s"] - a["t_s"])

    # t-half: first time the trace crosses baseline + delta/2
    half = baseline + delta / 2
    t_half = None
    for ti, v in zip(t, rb):
        if v >= half:
            t_half = ti
            break

    return {
        "n_points": len(rows),
        "duration_s": round(duration, 1),
        "baseline": round(baseline, 3),
        "plateau": round(plateau, 3),
        "delta": round(delta, 3),
        "slope_per_min": round(slope_per_min, 4),
        "auc": round(auc, 1),
        "t_half_s": t_half,
    }


def viability_call(m: dict, control_delta: float | None = None) -> dict:
    """Turn metrics into a viability interpretation.

    A large Δ (strong reduction) = metabolically active = viable cells. With a
    vehicle-control Δ supplied, we express it as a % of control viability.
    """
    if control_delta and control_delta > 0:
        pct = round(100 * m["delta"] / control_delta, 0)
        if pct >= 80:
            verdict = "high viability — little to no kill"
        elif pct >= 40:
            verdict = "partial viability — moderate kill"
        else:
            verdict = "low viability — strong kill"
        return {"viability_pct_of_control": pct, "verdict": verdict}
    # no control: qualitative, from the size of the reduction
    if m["delta"] >= 0.15:
        verdict = "strong dye reduction — cells metabolically active (viable)"
    elif m["delta"] >= 0.05:
        verdict = "weak dye reduction — reduced metabolic activity"
    else:
        verdict = "flat trace — little reduction, consistent with dead/quiescent cells"
    return {"viability_pct_of_control": None, "verdict": verdict}


def evidence_record(m: dict, call: dict, *, hypothesis: str, drug: str,
                    cell: str, readout: str) -> dict:
    """A Benchmate-ingestible record tying the assay to a hypothesis."""
    pct = call.get("viability_pct_of_control")
    if pct is not None and pct >= 80:
        interpretation = (f"Cells stayed ~{pct:.0f}% viable under {drug} — the "
                          "treatment did not kill them. This WEAKENS the hypothesis.")
        direction = "down-weight"
    elif pct is not None and pct < 40:
        interpretation = (f"Viability dropped to ~{pct:.0f}% of control under {drug} "
                          "— a strong kill. This SUPPORTS the hypothesis.")
        direction = "up-weight"
    else:
        interpretation = (f"{call['verdict']}. Compare against a vehicle control "
                          "to call support/refute quantitatively.")
        # The middle band, 40 to 80% of control, is a real partial effect.
        # Calling it "needs-control" was misleading once a control had been
        # supplied: nothing is missing, the answer is just equivocal, and a
        # partial kill should not shift a ranking either way on its own.
        direction = ("inconclusive" if pct is not None else "needs-control")
    return {
        "source": "alamarBlue viability assay (benchmate-demo rig)",
        "hypothesis_label": hypothesis,
        "drug": drug,
        "cell_line": cell,
        "readout": readout,
        "metrics": m,
        "viability": call,
        "interpretation": interpretation,
        "direction_for_benchmate": direction,
    }


def summarize(rec: dict) -> str:
    """One-paragraph evidence string for an agent prompt (mirrors the
    Geneformer context block)."""
    m = rec["metrics"]
    return (
        f"BENCH ASSAY EVIDENCE — {rec['source']}.\n"
        f"Hypothesis: {rec['hypothesis_label']}. {rec['drug']} on {rec['cell_line']}.\n"
        f"Readout: {rec['readout']}. red/blue {m['baseline']} → {m['plateau']} "
        f"(Δ {m['delta']}, slope {m['slope_per_min']}/min, t½ {m['t_half_s']}s).\n"
        f"Result: {rec['viability']['verdict']}.\n"
        f"Interpretation: {rec['interpretation']} "
        f"(suggested action: {rec['direction_for_benchmate']} this hypothesis)."
        + (f"\n{_audit.audit_summary(rec['audit'])}" if rec.get("audit") else "")
    )


def ingest(csv_path: str | Path, *, hypothesis: str, drug: str, cell: str,
           readout: str, control_delta: float | None = None) -> dict:
    """Full pipeline: read → metrics → call → evidence record (written to rig/).

    A whole-plate file handed to this function is delegated to `ingest_plate`
    rather than read as one trace. `read_run` ignores the condition column, so
    a 37-condition plate came back as a single 13,431-point series whose
    baseline was the average of every well and whose "discontinuity at t=0" was
    just the jump between one condition and the next. The agents then correctly
    reported that there was no vehicle control to normalise against, because
    from their side there was only one trace. Routing here means the caller
    cannot get it wrong.
    """
    if has_conditions(csv_path):
        return ingest_plate(csv_path, hypothesis=hypothesis, drug=drug,
                            cell=cell, readout=readout)
    rows = read_run(csv_path)
    m = metrics(rows)
    call = viability_call(m, control_delta)
    rec = evidence_record(m, call, hypothesis=hypothesis, drug=drug,
                          cell=cell, readout=readout)
    # Audit guard: an artifact must not move the ranking.
    a = _audit.audit_run(rows)
    rec["audit"] = a
    if a["severe"]:
        rec["direction_for_benchmate"] = "needs-recheck"
        rec["interpretation"] = ("ARTIFACT FLAGGED by the audit — this run isn't "
                                 "trustworthy and will not move any ranking until "
                                 "re-run. " + rec["interpretation"])
    RIG_DIR.mkdir(parents=True, exist_ok=True)
    out = RIG_DIR / f"{hypothesis}_assay.json"
    out.write_text(json.dumps(rec, indent=2))
    rec["_path"] = str(out)
    return rec


def available_assays() -> list[str]:
    """Hypothesis labels with a cached bench-assay record (mirrors
    available_geneformer_genes)."""
    if not RIG_DIR.exists():
        return []
    return sorted(p.stem.replace("_assay", "") for p in RIG_DIR.glob("*_assay.json"))


def assay_evidence(label: str) -> dict | None:
    """Load a cached assay evidence record by hypothesis label, or None."""
    p = RIG_DIR / f"{label}_assay.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Ingest an alamarBlue run into Benchmate evidence.")
    ap.add_argument("csv", help="path to the run CSV (t_s, R, G, B, red_blue)")
    ap.add_argument("--hypothesis", default="assay_run", help="hypothesis label to tie to")
    ap.add_argument("--drug", default="(unspecified compound)")
    ap.add_argument("--cell", default="(unspecified cell line)")
    ap.add_argument("--readout", default="alamarBlue red/blue reduction kinetics")
    ap.add_argument("--control-delta", type=float, default=None,
                    help="vehicle-control Δ(red_blue) to express viability as % of control")
    args = ap.parse_args()

    rec = ingest(args.csv, hypothesis=args.hypothesis, drug=args.drug,
                 cell=args.cell, readout=args.readout, control_delta=args.control_delta)
    print(summarize(rec))
    print(f"\n-> wrote {rec['_path']}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# A whole plate in one file
# ---------------------------------------------------------------------------
# The rig reads one well at a time, so the original format was a single trace
# and you typed the vehicle control's delta in by hand. That works, and it is
# also a step where a demo goes wrong and where a real user transcribes the
# wrong number. If the CSV carries a `condition` column, every arm and its
# control travel together and the normalisation happens here.

def has_conditions(path: str | Path) -> bool:
    """True if the CSV distinguishes conditions, so it holds a whole plate."""
    try:
        with open(path, newline="") as f:
            names = [c.lower().strip() for c in (csv.DictReader(f).fieldnames or [])]
        return any(c in names for c in ("condition", "sample", "well", "group",
                                        "treatment", "label"))
    except Exception:
        return False


def read_plate(path: str | Path) -> dict[str, list[dict]]:
    """Parse a multi-condition run into {condition: [{t_s, red_blue, R, G, B}]}.

    Replicates of the same condition are averaged per timepoint, which is what
    you want for a kinetic trace: three noisy wells give one usable curve.
    """
    C = ("condition", "sample", "well", "group", "treatment", "label")
    by_cond: dict[str, dict[float, list[dict]]] = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        cc = next((cols[k] for k in C if k in cols), None)
        if not cc:
            raise ValueError("No condition column; use read_run for a single trace.")
        for r in rd:
            try:
                cond = (r[cc] or "").strip() or "unlabelled"
                t = float(r["t_s"])
                by_cond.setdefault(cond, {}).setdefault(t, []).append({
                    "t_s": t,
                    "red_blue": float(r["red_blue"]),
                    "R": float(r.get("R", "nan") or "nan"),
                    "G": float(r.get("G", "nan") or "nan"),
                    "B": float(r.get("B", "nan") or "nan"),
                })
            except (KeyError, TypeError, ValueError):
                continue

    out: dict[str, list[dict]] = {}
    for cond, per_t in by_cond.items():
        rows = []
        for t in sorted(per_t):
            reps = per_t[t]
            def mean(k):
                vals = [x[k] for x in reps if x[k] == x[k]]   # drop NaN
                return sum(vals) / len(vals) if vals else float("nan")
            rows.append({"t_s": t, "red_blue": mean("red_blue"),
                         "R": mean("R"), "G": mean("G"), "B": mean("B"),
                         "n_replicates": len(reps)})
        if rows:
            out[cond] = rows
    if not out:
        raise ValueError("No usable rows (need t_s, red_blue, and a condition).")
    return out


def _is_blank(label: str) -> bool:
    """True if this condition is a cell-free blank rather than a treatment arm.

    A blank is medium plus dye with no cells in it. Nothing reduces the dye, so
    whatever delta it shows is drift: evaporation, thermal shift, sensor creep.
    Scoring it as a treatment arm produces the nonsense of a well with no cells
    reported as a strong kill.
    """
    t = label.lower()
    return any(n in t for n in ("blank", "no cells", "no-cell", "cell-free",
                                "cell free", "medium only", "media only"))


def _plate_control(conds) -> str | None:
    """The vehicle control, by the names people actually use for it."""
    for needle in ("vehicle", "dmso", "untreated", "no drug", "control"):
        for c in conds:
            if needle in c.lower() and not _is_blank(c):
                return c
    return None


def analyse_plate(by_cond: dict[str, list[dict]]) -> dict:
    """Metrics, audit and viability for every condition, normalised in-file.

    Blank wells, if the plate has any, are subtracted from every delta before
    normalisation. That is the step the printed protocol promises, and without
    it the blank's own drift inflates every viability on the plate.
    """
    stats: dict[str, dict] = {}
    for cond, rows in by_cond.items():
        m = metrics(rows)
        a = _audit.audit_run(rows)
        stats[cond] = {"metrics": m, "audit": a, "is_blank": _is_blank(cond),
                       "n_replicates": rows[0].get("n_replicates", 1)}

    # Blank correction. Average the blanks, since a single cell-free well is as
    # noisy as any other single well.
    blanks = [c for c, s in stats.items() if s["is_blank"]]
    blank_delta = None
    if blanks:
        usable = [stats[c]["metrics"]["delta"] for c in blanks
                  if not stats[c]["audit"]["severe"]]
        if usable:
            blank_delta = sum(usable) / len(usable)

    def corrected(cond: str) -> float:
        d = stats[cond]["metrics"]["delta"]
        return d - blank_delta if blank_delta is not None else d

    ctrl = _plate_control(stats)
    # A control with a severe artifact cannot normalise anything.
    ctrl_delta = None
    if ctrl and not stats[ctrl]["audit"]["severe"]:
        ctrl_delta = corrected(ctrl)

    for cond, s in stats.items():
        if s["is_blank"]:
            # Report the blank's drift so it is auditable, and do not pretend
            # it has a viability.
            s["viability"] = {
                "verdict": "blank (no cells)", "viability_pct_of_control": None,
                "delta": s["metrics"]["delta"],
                "note": ("Cell-free well. Its delta is drift, and it has been "
                         "subtracted from every other condition on this plate."),
            }
            continue
        m = dict(s["metrics"])
        m["delta"] = corrected(cond)
        s["metrics"]["delta_blank_corrected"] = round(m["delta"], 4)
        s["viability"] = viability_call(m, ctrl_delta)

    return {"conditions": stats, "control": ctrl, "control_delta": ctrl_delta,
            "blank_delta": round(blank_delta, 4) if blank_delta is not None else None,
            "blanks": blanks}


# Plate labels are abbreviated and design text is not. "Bortezomib 10 nM +
# chloroquine 10 uM" has to match "BTZ 10nM + CQ 10uM", so match on drug
# identity and on dose, not on shared words.
_SYNONYMS = {
    "bortezomib": {"bortezomib", "btz", "ps-341", "ps341", "velcade"},
    "chloroquine": {"chloroquine", "cq", "cqd"},
    "hydroxychloroquine": {"hydroxychloroquine", "hcq"},
    "kifunensine": {"kifunensine", "kif"},
    "carfilzomib": {"carfilzomib", "cfz", "kyprolis"},
    "bafilomycin": {"bafilomycin", "baf", "bafa1"},
    "staurosporine": {"staurosporine", "sts"},
    "thapsigargin": {"thapsigargin", "tg"},
    "mg132": {"mg132", "mg-132"},
    "eeyarestatin": {"eeyarestatin", "esi", "eer1"},
}
_DOSE_RE = re.compile(r"([\d.]+)\s*(pm|nm|um|µm|mm)\b", re.I)


def _drugs_in(text: str) -> set[str]:
    t = str(text or "").lower()
    return {canon for canon, alts in _SYNONYMS.items()
            if any(re.search(rf"\b{re.escape(a)}\b", t) for a in alts)}


def _dose_set(text: str) -> set[float]:
    """Doses in nM, so 10 uM and 10000 nM compare equal."""
    scale = {"pm": 1e-3, "nm": 1.0, "um": 1e3, "µm": 1e3, "mm": 1e6}
    return {round(float(m.group(1)) * scale[m.group(2).lower()], 4)
            for m in _DOSE_RE.finditer(str(text or ""))}


def _match_condition(conds, drug: str) -> str | None:
    """The condition that best matches the design's treatment string.

    The design already says what is being tested, so use it rather than
    assuming the most-killed arm is the interesting one. In a re-sensitisation
    design the most-killed arm is usually the sensitive control line.

    Scored on drugs matched first, then doses, so "BTZ 10nM + CQ 10uM" beats
    "BTZ 10nM" for a design that names both compounds.
    """
    want_drugs = _drugs_in(drug)
    want_doses = _dose_set(drug)
    if not want_drugs:
        return None
    best, best_key = None, (0, 0)
    for c in conds:
        if _plate_control([c]) == c:          # never pick the vehicle
            continue
        key = (len(want_drugs & _drugs_in(c)), len(want_doses & _dose_set(c)))
        if key > best_key:
            best, best_key = c, key
    # require every named compound to be present, so a single-agent arm is not
    # chosen for a combination design
    return best if best_key[0] >= len(want_drugs) else None


def ingest_plate(csv_path, *, hypothesis: str, drug: str, cell: str,
                 readout: str) -> dict:
    """Read a whole-plate CSV and file it, normalising against its own control."""
    by_cond = read_plate(csv_path)
    res = analyse_plate(by_cond)
    stats = res["conditions"]

    headline = _match_condition(stats, drug)
    lines = {c.split(" ")[0] for c in stats}
    if headline is None and len(lines) == 1:
        scored = [(k, v) for k, v in stats.items()
                  if k != res["control"]
                  and v["viability"].get("viability_pct_of_control") is not None]
        if scored:
            headline = min(
                scored, key=lambda kv: kv[1]["viability"]["viability_pct_of_control"])[0]

    if headline:
        h = stats[headline]
        rec = evidence_record(h["metrics"], h["viability"], hypothesis=hypothesis,
                              drug=f"{drug} [{headline}]", cell=cell,
                              readout=readout)
        rec["audit"] = h["audit"]
        if h["audit"]["severe"]:
            rec["direction_for_benchmate"] = "needs-recheck"
            rec["interpretation"] = ("ARTIFACT FLAGGED by the audit on this arm, "
                                     "so it will not move any ranking until "
                                     "re-run. " + rec["interpretation"])
    else:
        first = next(iter(stats.values()))
        rec = evidence_record(first["metrics"], first["viability"],
                              hypothesis=hypothesis, drug=drug, cell=cell,
                              readout=readout)
        rec["audit"] = {"severe": False, "flags": []}
        rec["direction_for_benchmate"] = "no-change"
        rec["interpretation"] = (
            f"This plate covers more than one cell line "
            f"({', '.join(sorted(lines))}) and no condition clearly matched the "
            f"treatment in the design, so scoring one arm would be a guess. The "
            f"per-condition table is the evidence; the feedback step reads it "
            f"against your design.")

    rec["assay"] = "viability-plate"
    rec["headline_condition"] = headline
    rec["control"] = res["control"]
    rec["control_delta"] = res["control_delta"]
    rec["conditions"] = {
        c: {"delta": s["metrics"]["delta"],
            "t_half_s": s["metrics"].get("t_half_s"),
            "viability_pct": s["viability"].get("viability_pct_of_control"),
            "verdict": s["viability"]["verdict"],
            "n_replicates": s["n_replicates"],
            "audit_flags": s["audit"]["flags"]}
        for c, s in stats.items()}

    RIG_DIR.mkdir(parents=True, exist_ok=True)
    out = RIG_DIR / f"{hypothesis}_plate.json"
    out.write_text(json.dumps(rec, indent=2))
    rec["_path"] = str(out)
    return rec


def summarize_plate(rec: dict) -> str:
    """The prompt block the feedback step reads for a whole plate."""
    lines = [
        f"alamarBlue viability plate on {rec['cell_line']}.",
        f"Treatment under test: {rec['drug']}.",
        f"Vehicle control: {rec.get('control')} "
        f"(delta {rec.get('control_delta')}).",
        f"Verdict on the tested arm: {rec['viability']['verdict']}.",
        rec["interpretation"], "",
        "Every condition on the plate, viability as % of the vehicle control:",
    ]
    rows = sorted(rec["conditions"].items(),
                  key=lambda kv: (kv[1]["viability_pct"] is None,
                                  kv[1]["viability_pct"] or 0))
    for cond, s in rows:
        v = s["viability_pct"]
        lines.append(
            f"  {cond}: {'control' if v is None else str(round(v)) + '%'}"
            f", delta {s['delta']}, n={s['n_replicates']}"
            + (f"  [AUDIT: {'; '.join(s['audit_flags'])}]"
               if s["audit_flags"] else ""))
    return "\n".join(lines)
