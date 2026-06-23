"""Discrimination experiment: does ontology grounding catch the false hypotheses
WITHOUT punishing the novel ones?

Runs the fair judge over the adversarial gold set twice — grounding OFF vs ON —
and reports, per hypothesis kind, how the ranking moved. The win condition
(motivated by Wei-Hung Weng's feedback) is:

    trap_demotion   > 0     grounding pushed the fluent-but-FALSE traps DOWN     (good)
    novelty_penalty ~ 0     grounding did NOT push the NOVEL-but-true ones down  (good)

If novelty_penalty is large and positive, grounding is behaving like the
consensus-filter we were warned about — penalising paradigm-shifting ideas.

Live (needs ANTHROPIC_API_KEY) and needs OntoMCP running for the ON arm:
    python -m benchmark.run_adversarial --cycles 6 --n-per-cycle 8
"""
from __future__ import annotations

import argparse
import os
import statistics
from collections import defaultdict

from benchmark.gold_set_adversarial import GOLD_ADV, adversarial_hypotheses
from benchmark.metrics import spearman


def _ranks(hyps) -> dict[int, int]:
    """gold_index -> rank (1 = best by Elo)."""
    order = sorted(hyps, key=lambda h: -h.elo)
    return {h.meta["gold"]: i + 1 for i, h in enumerate(order)}


def summarize(off_hyps, on_hyps) -> dict:
    """Compare the two finished runs. Pure function — unit-testable on mocks."""
    roff, ron = _ranks(off_hyps), _ranks(on_hyps)
    kind_of = {h.meta["gold"]: h.meta["kind"] for h in off_hyps}

    buckets: dict[str, dict[str, list]] = defaultdict(lambda: {"off": [], "on": []})
    for g, kind in kind_of.items():
        buckets[kind]["off"].append(roff[g])
        buckets[kind]["on"].append(ron[g])

    by_kind = {}
    for kind, d in buckets.items():
        mo, mn = statistics.mean(d["off"]), statistics.mean(d["on"])
        by_kind[kind] = {"mean_rank_off": mo, "mean_rank_on": mn,
                         "delta": mn - mo}  # +ve = moved to WORSE ranks under grounding

    def sp(hyps):
        return spearman([-h.meta["gold"] for h in hyps], [h.elo for h in hyps])

    return {
        "by_kind": by_kind,
        "trap_demotion": by_kind.get("trap", {}).get("delta", 0.0),
        "novelty_penalty": by_kind.get("novel", {}).get("delta", 0.0),
        "spearman_off": sp(off_hyps),
        "spearman_on": sp(on_hyps),
    }


def _print_report(res: dict, off_hyps, on_hyps) -> None:
    print("\n" + "=" * 68)
    print("ONTOLOGY DISCRIMINATION TEST  (grounding OFF vs ON)")
    print("=" * 68)
    print(f"{'kind':8}  {'mean rank OFF':>13}  {'mean rank ON':>12}  {'Δ rank':>7}")
    for kind in ("solid", "novel", "trap"):
        r = res["by_kind"].get(kind)
        if r:
            print(f"{kind:8}  {r['mean_rank_off']:>13.1f}  {r['mean_rank_on']:>12.1f}"
                  f"  {r['delta']:>+7.1f}")
    print("-" * 68)
    print(f"  trap demotion   (want > 0, traps sink)        : {res['trap_demotion']:+.1f}")
    print(f"  novelty penalty (want ~ 0, novel stays put)   : {res['novelty_penalty']:+.1f}")
    print(f"  spearman vs gold   OFF {res['spearman_off']:+.2f}  ->  ON {res['spearman_on']:+.2f}")
    print("-" * 68)
    good = res["trap_demotion"] > 0 and res["novelty_penalty"] <= 0.75
    if good:
        print("  ✓ Grounding caught the traps without punishing novelty.")
    elif res["trap_demotion"] <= 0:
        print("  ✗ Grounding did not demote the traps. Coverage/placement issue.")
    else:
        print("  ⚠ Grounding demoted traps BUT also dragged novel ideas down —")
        print("    the consensus-filter failure mode. Move grounding to review-only.")
    print("  (one run is one noisy sample — run 3+ times and read the median)\n")

    print("Final ON leaderboard (where did each kind land?):")
    for rank, h in enumerate(sorted(on_hyps, key=lambda h: -h.elo), 1):
        print(f"  {rank}. [{h.elo:6.1f}] ({h.meta['kind']:5}) {h.statement[:64]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--n-per-cycle", type=int, default=8)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this live experiment.")

    from benchmark.run_benchmark import run_tournament, _fair_judge_fn

    print(f"Ranking {len(GOLD_ADV)} adversarial hypotheses, grounding OFF vs ON "
          f"({args.cycles}×{args.n_per_cycle} matches each)…")
    off = adversarial_hypotheses()
    run_tournament(off, _fair_judge_fn(ontology=False), args.cycles, args.n_per_cycle)
    on = adversarial_hypotheses()
    run_tournament(on, _fair_judge_fn(ontology=True), args.cycles, args.n_per_cycle)

    _print_report(summarize(off, on), off, on)


if __name__ == "__main__":
    main()
