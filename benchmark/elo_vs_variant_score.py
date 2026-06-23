"""Does the Elo ranking agree with an independent quantitative predictor?

Wei-Hung Weng's note: before trusting Elo to pick wet-lab candidates, check how
well it correlates with an external sequence-to-function score (AlphaGenome /
Enformer). If the correlation is weak, Elo alone is the wrong selection tool —
and your benchmarking just surfaced that.

This is intentionally simple: it reads a JSON file pairing each hypothesis's Elo
with its variant score, and reports the Spearman correlation. Building that file
is the manual part (see WORKFLOW below).

WORKFLOW
  1. Take the hypotheses you can frame as a regulatory variant (a gene + a
     specific change). Perturbation-only hypotheses don't apply.
  2. Score each variant with co_scientist/variant_scorer.py (AlphaGenome API is
     easiest; Enformer on Colab otherwise).
  3. Save a file like benchmark/variant_scores.json:
       [{"label": "H_HRD1_promoter", "elo": 1283.0, "score": 0.42}, ...]
     ("elo" = that hypothesis's Elo from a tournament / your state.json.)
  4. Run:  python -m benchmark.elo_vs_variant_score --scores benchmark/variant_scores.json
     Or try the shape first:  python -m benchmark.elo_vs_variant_score --demo
"""
from __future__ import annotations

import argparse
import json
import random

from benchmark.metrics import spearman


def correlate(elo: list[float], score: list[float]) -> dict:
    """Spearman between Elo and the independent score. Pure + testable."""
    n = len(elo)
    if n != len(score):
        raise ValueError("elo and score must be the same length")
    if n < 3:
        return {"n": n, "spearman": None,
                "note": "need >=3 paired hypotheses to correlate"}
    rho = spearman(elo, score)
    if rho >= 0.6:
        verdict = ("strong agreement — Elo tracks the independent predictor; "
                   "using Elo to shortlist candidates looks defensible.")
    elif rho >= 0.3:
        verdict = ("weak/partial agreement — Elo and the model only loosely "
                   "agree; don't rely on Elo alone for final candidate choice.")
    else:
        verdict = ("little to no agreement — Elo and the independent predictor "
                   "disagree. This is exactly the gap to flag: the LLM ranking "
                   "is not capturing what the sequence model sees.")
    return {"n": n, "spearman": rho, "verdict": verdict}


def _load(path: str) -> tuple[list[float], list[float], list[str]]:
    rows = json.loads(open(path).read())
    elo = [float(r["elo"]) for r in rows]
    score = [float(r["score"]) for r in rows]
    labels = [str(r.get("label", i)) for i, r in enumerate(rows)]
    return elo, score, labels


def _demo() -> tuple[list[float], list[float], list[str]]:
    """Synthetic data so you can see the output shape before scoring anything."""
    rng = random.Random(7)
    elo, score, labels = [], [], []
    for i in range(8):
        e = 1200 + rng.uniform(-80, 80)
        # deliberately weak coupling, so the demo shows a low correlation
        s = 0.3 + 0.0008 * (e - 1200) + rng.uniform(-0.15, 0.15)
        elo.append(round(e, 1)); score.append(round(s, 3)); labels.append(f"H{i}")
    return elo, score, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", help="JSON file: [{label, elo, score}, ...]")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    args = ap.parse_args()

    if args.demo or not args.scores:
        if not args.demo:
            print("No --scores file given; showing --demo output.\n")
        elo, score, labels = _demo()
    else:
        elo, score, labels = _load(args.scores)

    res = correlate(elo, score)
    print("=" * 60)
    print("ELO  vs  INDEPENDENT VARIANT SCORE")
    print("=" * 60)
    for l, e, s in sorted(zip(labels, elo, score), key=lambda t: -t[1]):
        print(f"  {l:24}  Elo {e:7.1f}   score {s:.3f}")
    print("-" * 60)
    if res["spearman"] is None:
        print(f"  {res['note']}")
    else:
        print(f"  n = {res['n']}   Spearman(Elo, score) = {res['spearman']:+.2f}")
        print(f"  {res['verdict']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
