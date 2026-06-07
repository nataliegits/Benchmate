"""Benchmate benchmarking CLI.

  python -m benchmark.run_benchmark simulate            # free, no API
  python -m benchmark.run_benchmark judge-eval          # live: is the judge good?
  python -m benchmark.run_benchmark validate            # live: does ranking match gold?
  python -m benchmark.run_benchmark compare             # live: fair judge vs naive judge

Only `simulate` is free. The others make real API calls on the small gold set
in gold_set.py; keep --matches modest while iterating.
"""
from __future__ import annotations

import argparse
import os
from typing import Callable

from co_scientist.elo import schedule_matches, update_elo
from co_scientist.state import Hypothesis, INITIAL_ELO

from .gold_set import GOLD, GOLD_RANK, gold_hypotheses
from .metrics import spearman, topk_jaccard

CRITERIA = ["novelty", "mechanistic plausibility", "falsifiability of the experiment"]


# ----- a tournament runner parameterised by the judge ---------------------

def run_tournament(hyps: list[Hypothesis], judge: Callable[[Hypothesis, Hypothesis], str],
                   cycles: int, n_per_cycle: int) -> None:
    """Run the real schedule_matches/update_elo loop, deciding each match with
    `judge(a, b) -> 'A' | 'B' | 'draw'`. Mutates hyps' Elo in place."""
    for h in hyps:
        h.elo = float(INITIAL_ELO)
        h.matches_played = 0
    for _ in range(cycles):
        for a, b in schedule_matches(hyps, n_matches=n_per_cycle):
            w = judge(a, b)
            if w == "A":
                update_elo(a, b)
            elif w == "B":
                update_elo(b, a)
            else:
                update_elo(a, b, draw=True)


def _report_vs_gold(hyps: list[Hypothesis], label: str) -> float:
    """Score a finished tournament against the gold ranking."""
    # Each hyp was built in gold order, so its gold rank = its build index.
    gold_rank_of = {h.id: GOLD_RANK[i] for i, h in enumerate(hyps)}
    # spearman wants two aligned score lists; use (−gold_rank) and elo
    gold_scores = [-gold_rank_of[h.id] for h in hyps]   # higher = better
    elo_scores = [h.elo for h in hyps]
    rho = spearman(gold_scores, elo_scores)

    final = sorted(hyps, key=lambda h: -h.elo)
    gold_order = [h.id for h in sorted(hyps, key=lambda h: gold_rank_of[h.id])]
    final_order = [h.id for h in final]
    top1 = final_order[0] == gold_order[0]
    top3 = topk_jaccard(final_order, gold_order, 3)

    print(f"\n  [{label}]")
    print(f"    spearman vs gold : {rho:+.2f}   (1.0 = matches gold tier order)")
    print(f"    top-1 correct    : {'yes' if top1 else 'no '}   "
          f"(gold best = a tier-A hypothesis)")
    print(f"    top-3 overlap    : {top3:.0%}")
    print("    leaderboard:")
    for rank, h in enumerate(final, 1):
        tier = GOLD[next(i for i, x in enumerate(hyps) if x.id == h.id)]["tier"]
        print(f"      {rank}. [{h.elo:6.1f}] (tier {tier}) {h.statement[:72]}")
    return rho


# ----- judges --------------------------------------------------------------

def _fair_judge_fn():
    from .fair_judge import judge_pair
    return lambda a, b: judge_pair(a, b, CRITERIA, role="ranking").winner


def _naive_judge_fn():
    from .judge_eval import _naive_winner
    return lambda a, b: _naive_winner(a, b)


def _require_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run live benchmarks "
                         "(only `simulate` is free).")


# ----- subcommands ---------------------------------------------------------

def cmd_simulate(_args):
    from .simulate import main
    main()


def cmd_judge_eval(args):
    _require_key()
    from .judge_eval import evaluate_judge, print_report
    print_report(evaluate_judge(max_pairs=args.max_pairs, role=args.role), args.role)


def cmd_validate(args):
    _require_key()
    hyps = gold_hypotheses()
    run_tournament(hyps, _fair_judge_fn(), cycles=args.cycles,
                   n_per_cycle=args.n_per_cycle)
    _report_vs_gold(hyps, f"fair judge · {args.cycles}×{args.n_per_cycle} matches")


def cmd_compare(args):
    _require_key()
    print("Ranking the same gold set under two judges "
          f"({args.cycles}×{args.n_per_cycle} matches each):")
    h1 = gold_hypotheses()
    run_tournament(h1, _naive_judge_fn(), args.cycles, args.n_per_cycle)
    rho_naive = _report_vs_gold(h1, "NAIVE judge (current: fixed order, thin prompt)")
    h2 = gold_hypotheses()
    run_tournament(h2, _fair_judge_fn(), args.cycles, args.n_per_cycle)
    rho_fair = _report_vs_gold(h2, "FAIR judge (order-swapped, richer prompt)")
    print(f"\n  Δ spearman (fair − naive) = {rho_fair - rho_naive:+.2f}  "
          f"{'(fair judge ranks closer to gold)' if rho_fair > rho_naive else ''}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmate benchmarking toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("simulate", help="free Monte Carlo over the real Elo code")

    je = sub.add_parser("judge-eval", help="live: accuracy/bias/consistency of the judge")
    je.add_argument("--max-pairs", type=int, default=8)
    je.add_argument("--role", default="ranking")

    va = sub.add_parser("validate", help="live: rank the gold set, score vs gold order")
    va.add_argument("--cycles", type=int, default=6)
    va.add_argument("--n-per-cycle", type=int, default=8)

    cm = sub.add_parser("compare", help="live: fair judge vs naive judge on the gold set")
    cm.add_argument("--cycles", type=int, default=6)
    cm.add_argument("--n-per-cycle", type=int, default=8)
    return p


def main():
    args = build_parser().parse_args()
    {"simulate": cmd_simulate, "judge-eval": cmd_judge_eval,
     "validate": cmd_validate, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
