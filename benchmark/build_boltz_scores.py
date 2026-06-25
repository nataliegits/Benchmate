"""Produce benchmark/boltz_scores.json for the Boltz cross-check.

Ranks the binding-hypothesis gold set with the LLM judge (Elo), scores each
protein+ligand pair with Boltz, and writes the merged
[{label, elo, score}, ...] that the "Cross-check with other models" tab and
elo_vs_variant_score read.

Unlike AlphaGenome, Boltz is a plain API (works on your Python 3.9), so no Colab:
this one script does everything.

Needs:
  ANTHROPIC_API_KEY   — for the judge (Elo)
  BOLTZ_API_KEY       — for Boltz scoring   (sign up + $100 credits, code
                        BOLTZLAUNCH, at https://api.boltz.bio/console/signup)

If BOLTZ_API_KEY isn't set, it falls back to a hand-made
benchmark/boltz_raw_scores.json ({label: score}) if you have one.

    python -m benchmark.build_boltz_scores
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from co_scientist.state import Hypothesis
from benchmark.gold_set_binding import GOLD_BINDING, binding_targets

HERE = Path(__file__).resolve().parent
RAW_IN = HERE / "boltz_raw_scores.json"     # optional fallback {label: score}
OUT = HERE / "boltz_scores.json"


def _elo_by_label(cycles: int, n_per_cycle: int) -> dict[str, float]:
    from benchmark.run_benchmark import run_tournament, _fair_judge_fn
    hyps, label_of = [], {}
    for g in GOLD_BINDING:
        h = Hypothesis.new(statement=g["statement"], rationale=g["rationale"],
                           experiment="")
        label_of[h.id] = g["label"]
        hyps.append(h)
    run_tournament(hyps, _fair_judge_fn(ontology=False), cycles, n_per_cycle)
    return {label_of[h.id]: round(h.elo, 1) for h in hyps}


def _boltz_scores() -> dict[str, float]:
    """Score each target with Boltz, or fall back to a hand-made raw file."""
    from co_scientist.boltz_scorer import score_binding, boltz_available
    if boltz_available():
        out = {}
        for t in binding_targets():
            print(f"  scoring {t.label} with Boltz…")
            s = score_binding(t)
            if s is not None:
                out[t.label] = s
        return out
    if RAW_IN.exists():
        print(f"BOLTZ_API_KEY not set — using {RAW_IN.name}.")
        return json.loads(RAW_IN.read_text())
    raise SystemExit("Set BOLTZ_API_KEY (api.boltz.bio) or provide "
                     f"{RAW_IN.name} ({{label: score}}).")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY — the judge ranks the hypotheses "
                         "to produce the Elo column.")

    print("Scoring binding hypotheses with Boltz…")
    boltz = _boltz_scores()
    print(f"Got {len(boltz)} Boltz scores. Ranking hypotheses for Elo…")
    elo = _elo_by_label(6, 8)

    rows = []
    for g in GOLD_BINDING:
        lab = g["label"]
        if lab in boltz and lab in elo:
            rows.append({"label": lab, "elo": elo[lab], "score": float(boltz[lab])})
        else:
            print(f"  (skipping {lab}: missing "
                  f"{'Boltz score' if lab not in boltz else 'Elo'})")

    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT.name} with {len(rows)} rows:")
    for r in rows:
        print(f"  {r['label']:18}  Elo {r['elo']:7.1f}   score {r['score']:.4f}")
    print("\nNow open the 'Cross-check with other models' tab → Boltz, or run:\n"
          f"  python -m benchmark.elo_vs_variant_score --scores benchmark/{OUT.name}")


if __name__ == "__main__":
    main()
