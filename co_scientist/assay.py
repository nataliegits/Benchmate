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
import json
from pathlib import Path

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
        direction = "needs-control"
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
    )


def ingest(csv_path: str | Path, *, hypothesis: str, drug: str, cell: str,
           readout: str, control_delta: float | None = None) -> dict:
    """Full pipeline: read → metrics → call → evidence record (written to rig/)."""
    rows = read_run(csv_path)
    m = metrics(rows)
    call = viability_call(m, control_delta)
    rec = evidence_record(m, call, hypothesis=hypothesis, drug=drug,
                          cell=cell, readout=readout)
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
