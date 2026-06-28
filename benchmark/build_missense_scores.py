"""Score the missense gold set with AlphaMissense, merged with Elo.

Ranks the real ClinVar missense variants with the LLM judge (Elo), scores each
with AlphaMissense (via Ensembl VEP), and writes the file the Cross-check tab
reads:
    benchmark/alphamissense_scores.json   [{label, elo, score, clinsig}, ...]

Because each variant carries its ClinVar classification, this also prints a
calibration check: mean AlphaMissense for pathogenic vs benign (they should
separate). Variants AlphaMissense can't score (no missense consequence) are
skipped.

    python -m benchmark.fetch_clinvar          # 1. get real variants
    python -m benchmark.build_missense_scores   # 2. rank + score + merge

Needs ANTHROPIC_API_KEY (Elo). Ensembl VEP is free, no key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from co_scientist.state import Hypothesis
from co_scientist.target_scorer import alphamissense_score
from benchmark.gold_set_missense import missense_hypotheses

HERE = Path(__file__).resolve().parent
OUT = HERE / "alphamissense_scores.json"


def _elo_by_label(hyps_spec, cycles: int, n_per_cycle: int) -> dict[str, float]:
    from benchmark.run_benchmark import run_tournament, _fair_judge_fn
    hyps, label_of = [], {}
    for g in hyps_spec:
        h = Hypothesis.new(statement=g["statement"], rationale=g["rationale"], experiment="")
        label_of[h.id] = g["label"]
        hyps.append(h)
    run_tournament(hyps, _fair_judge_fn(ontology=False), cycles, n_per_cycle)
    return {label_of[h.id]: round(h.elo, 1) for h in hyps}


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY — the judge ranks the variants for Elo.")
    spec = missense_hypotheses()
    print(f"Ranking {len(spec)} missense hypotheses for Elo…")
    elo = _elo_by_label(spec, 6, 8)

    print("\nAlphaMissense (Ensembl VEP) pathogenicity:")
    rows = []
    for g in spec:
        s = alphamissense_score(g["chrom"], g["pos"], g["ref"], g["alt"])
        if s is None:
            print(f"  {g['label']:28} no AlphaMissense score — skipping")
            continue
        rows.append({"label": g["label"], "elo": elo[g["label"]],
                     "score": float(s), "clinsig": g["clinsig"]})
        print(f"  {g['label']:28} AM {s:.3f}  (ClinVar {g['clinsig']}, Elo {elo[g['label']]})")

    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\n-> wrote {OUT.name} ({len(rows)} rows)")

    # calibration: do pathogenic variants score higher than benign?
    path = [r["score"] for r in rows if r["clinsig"] == "pathogenic"]
    benign = [r["score"] for r in rows if r["clinsig"] == "benign"]
    if path and benign:
        print(f"\nCalibration check — mean AlphaMissense:")
        print(f"  pathogenic: {sum(path)/len(path):.3f}  (n={len(path)})")
        print(f"  benign:     {sum(benign)/len(benign):.3f}  (n={len(benign)})")
        print(f"  separation: {sum(path)/len(path) - sum(benign)/len(benign):+.3f} "
              "(positive = AlphaMissense ranks pathogenic above benign, as it should)")
    print("\nThen: python -m benchmark.elo_vs_variant_score "
          "--scores benchmark/alphamissense_scores.json")


if __name__ == "__main__":
    main()
