"""Merge AlphaGenome scores (from Colab) with Elo, into variant_scores.json.

The full loop for the Elo-vs-predictor experiment:

  1. Score the variants in Colab  -> download alphagenome_scores.json
     (a {label: score} mapping) into this benchmark/ folder.
  2. Run THIS script. It ranks the variant-framed hypotheses with the live judge
     to get an Elo per hypothesis, merges in the AlphaGenome scores by label, and
     writes variant_scores.json = [{label, elo, score}, ...].
  3. Open the Benchmark tab section 7 (or `python -m benchmark.elo_vs_variant_score
     --scores benchmark/variant_scores.json`) to see the correlation.

Needs ANTHROPIC_API_KEY (for the judge). OntoMCP is optional here.

    python -m benchmark.build_variant_scores
    python -m benchmark.build_variant_scores --cycles 6 --n-per-cycle 8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from co_scientist.state import Hypothesis
from benchmark.gold_set_variants import GOLD_VARIANTS

HERE = Path(__file__).resolve().parent
SCORES_IN = HERE / "alphagenome_scores.json"
OUT = HERE / "variant_scores.json"


def _elo_by_label(cycles: int, n_per_cycle: int) -> dict[str, float]:
    """Rank the variant-framed hypotheses with the fair judge; return label -> Elo."""
    from benchmark.run_benchmark import run_tournament, _fair_judge_fn
    hyps, label_of = [], {}
    for g in GOLD_VARIANTS:
        h = Hypothesis.new(statement=g["statement"], rationale=g["rationale"],
                           experiment="")
        label_of[h.id] = g["label"]
        hyps.append(h)
    run_tournament(hyps, _fair_judge_fn(ontology=False), cycles, n_per_cycle)
    return {label_of[h.id]: round(h.elo, 1) for h in hyps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--n-per-cycle", type=int, default=8)
    args = ap.parse_args()

    if not SCORES_IN.exists():
        raise SystemExit(f"Missing {SCORES_IN.name}. Download it from the Colab "
                         "scoring notebook into the benchmark/ folder first.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY — the judge ranks the hypotheses "
                         "to produce the Elo column.")

    ag = json.loads(SCORES_IN.read_text())          # {label: score}
    print(f"Loaded {len(ag)} AlphaGenome scores. Ranking hypotheses for Elo…")
    elo = _elo_by_label(args.cycles, args.n_per_cycle)

    rows = []
    for g in GOLD_VARIANTS:
        lab = g["label"]
        if lab in ag and lab in elo:
            rows.append({"label": lab, "elo": elo[lab], "score": float(ag[lab])})
        else:
            print(f"  (skipping {lab}: missing "
                  f"{'AlphaGenome score' if lab not in ag else 'Elo'})")

    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT.name} with {len(rows)} rows:")
    for r in rows:
        print(f"  {r['label']:18}  Elo {r['elo']:7.1f}   score {r['score']:.4f}")
    print("\nNow run:  python -m benchmark.elo_vs_variant_score "
          f"--scores benchmark/{OUT.name}")


if __name__ == "__main__":
    main()
