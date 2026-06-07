"""Free, LLM-free Monte Carlo simulator for the Elo tournament.

Why this exists
---------------
You can't tune your Elo system by running the real loop — each run costs API
money and you only see one noisy outcome. This simulator replays your ACTUAL
`co_scientist.elo` code (schedule_matches + update_elo) against synthetic
hypotheses whose *true* quality we control, so we can ask, for free and
thousands of times:

  - With N hypotheses, how many matches until the ranking stops moving?
  - Does your current K-factor (40/20/10) help or hurt at this small scale?
  - How often does the true-best hypothesis actually end up ranked #1?
  - How stable is the leaderboard across repeated runs (the demo question:
    "if I click Run twice, do I get the same winner")?

Model
-----
Each hypothesis i has a latent quality q_i (evenly spaced, so there's a known
true ranking). When i and j play, i wins with probability

    p = 1 / (1 + exp(-slope * (q_i - q_j)))

`slope` is set from `judge_skill` = P(judge picks the better of two ADJACENT
hypotheses). judge_skill=0.5 is a coin flip (useless judge); 0.75 is decent.
`position_bias` optionally gives the first-listed hypothesis a constant edge,
to show how an order-biased judge corrupts the ranking.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable

import co_scientist.elo as elo_mod
from co_scientist.elo import schedule_matches, update_elo
from co_scientist.state import Hypothesis, INITIAL_ELO

from .metrics import spearman, rank_churn


# ----- K-factor schedules to compare (drop-in replacements for elo.k_factor)

def k_const(value: float) -> Callable[[int], float]:
    return lambda m: value


def k_schedule(thresholds: list[tuple[int, float]], tail: float) -> Callable[[int], float]:
    """thresholds=[(5,40),(15,20)], tail=10  ==  Benchmate's default."""
    def f(m: int) -> float:
        for limit, k in thresholds:
            if m < limit:
                return k
        return tail
    return f


CURRENT_K = k_schedule([(5, 40.0), (15, 20.0)], 10.0)   # your elo.py today


# ----- one simulated tournament -------------------------------------------

def _make_hypotheses(n: int) -> tuple[list[Hypothesis], dict[str, float], dict[str, int]]:
    """n hypotheses with evenly spaced true quality in [0, 1].

    Returns the hypotheses (all at INITIAL_ELO), a map id -> true quality, and
    a map id -> stable label (true-quality rank, 0 = best). Labels are stable
    across runs even though hypothesis ids are random, so we can compare
    leaderboards between independent replicates."""
    hyps, quality, label = [], {}, {}
    for i in range(n):
        h = Hypothesis.new(statement=f"hypothesis {i}", rationale="", experiment="")
        h.elo = float(INITIAL_ELO)
        h.matches_played = 0
        q = i / (n - 1) if n > 1 else 0.5
        hyps.append(h)
        quality[h.id] = q
        label[h.id] = (n - 1) - i          # i=n-1 is best quality -> label 0
    return hyps, quality, label


def _slope_from_skill(judge_skill: float, n: int) -> float:
    """slope such that two adjacent hypotheses (quality gap 1/(n-1)) are
    ordered correctly with probability judge_skill."""
    js = min(max(judge_skill, 0.5001), 0.9999)
    gap = 1.0 / (n - 1) if n > 1 else 1.0
    return math.log(js / (1 - js)) / gap


@dataclass
class RunResult:
    final_labels: list[int]           # true-quality labels in finishing order (0=best)
    spearman: float                   # final Elo ranking vs true ranking
    top1_correct: bool                # did the true best (label 0) finish #1?
    top3_contains_best: bool
    churn_curve: list[float]          # rank churn measured each cycle
    total_matches: int


def simulate_once(n: int = 6, cycles: int = 4, n_per_cycle: int = 6,
                  judge_skill: float = 0.72, position_bias: float = 0.0,
                  k_fn: Callable[[int], float] = CURRENT_K,
                  seed: int = 0) -> RunResult:
    random.seed(seed)
    orig_k = elo_mod.k_factor
    elo_mod.k_factor = k_fn                      # patch the real update_elo's K
    try:
        hyps, quality, label = _make_hypotheses(n)
        slope = _slope_from_skill(judge_skill, n)

        churn_curve: list[float] = []
        prev_order = [label[h.id] for h in sorted(hyps, key=lambda h: -h.elo)]
        total = 0
        for _ in range(cycles):
            for a, b in schedule_matches(hyps, n_matches=n_per_cycle):
                qa, qb = quality[a.id], quality[b.id]
                p = 1.0 / (1.0 + math.exp(-slope * (qa - qb)))
                p = min(1.0, p + position_bias)      # first-listed (current leader) edge
                if random.random() < p:
                    update_elo(a, b)
                else:
                    update_elo(b, a)
                total += 1
            now = [label[h.id] for h in sorted(hyps, key=lambda h: -h.elo)]
            churn_curve.append(rank_churn(prev_order, now))
            prev_order = now

        final_labels = [label[h.id] for h in sorted(hyps, key=lambda h: -h.elo)]
        final_scores = [h.elo for h in hyps]
        true_scores = [quality[h.id] for h in hyps]
        return RunResult(
            final_labels=final_labels,
            spearman=spearman(true_scores, final_scores),
            top1_correct=(final_labels[0] == 0),
            top3_contains_best=(0 in final_labels[:3]),
            churn_curve=churn_curve,
            total_matches=total,
        )
    finally:
        elo_mod.k_factor = orig_k


# ----- aggregate over many seeds ------------------------------------------

@dataclass
class Summary:
    label: str
    n: int
    matches_per_hyp: float
    mean_spearman: float
    sd_spearman: float
    top1_accuracy: float          # P(true best ranked #1)
    top3_accuracy: float          # P(true best in top 3)
    repeatability: float          # P(two independent runs pick the SAME #1)
    final_churn: float            # mean churn in the last cycle


def _repeatability(winner_labels: list[int]) -> float:
    """Probability two independent runs land on the same #1 = sum p_k^2.
    This is the demo question: 'if I click Run twice, same winner?'"""
    if not winner_labels:
        return float("nan")
    from collections import Counter
    total = len(winner_labels)
    return sum((c / total) ** 2 for c in Counter(winner_labels).values())


def evaluate(label: str, replicates: int = 300, **kw) -> Summary:
    runs = [simulate_once(seed=s, **kw) for s in range(replicates)]
    n = kw.get("n", 6)
    sp = [r.spearman for r in runs]
    return Summary(
        label=label, n=n,
        matches_per_hyp=runs[0].total_matches * 2 / n,
        mean_spearman=mean(sp), sd_spearman=pstdev(sp),
        top1_accuracy=mean(r.top1_correct for r in runs),
        top3_accuracy=mean(r.top3_contains_best for r in runs),
        repeatability=_repeatability([r.final_labels[0] for r in runs]),
        final_churn=mean(r.churn_curve[-1] for r in runs if r.churn_curve),
    )


def _fmt(s: Summary) -> str:
    return (f"  {s.label:<34} matches/hyp={s.matches_per_hyp:4.1f}  "
            f"spearman={s.mean_spearman:5.2f}±{s.sd_spearman:.2f}  "
            f"top1={s.top1_accuracy:4.0%}  top3={s.top3_accuracy:4.0%}  "
            f"repeat={s.repeatability:4.0%}  churn={s.final_churn:.2f}")


def main() -> None:
    print("=" * 100)
    print("BENCHMATE ELO SIMULATOR  —  free, runs against your real elo.py")
    print("=" * 100)
    print("Reading: spearman→ranking accuracy vs ground truth (1.0=perfect),  "
          "top1→true best ends #1,\n         repeat→two runs pick the same "
          "winner,  churn→rank movement in the final cycle (→0 = converged)\n")

    N = 6
    skill = 0.70

    print(f"[1] MATCH BUDGET  (n={N} hypotheses, judge_skill={skill:.0%}, "
          f"current K=40/20/10)")
    print("    Your state.json today shows ~2 matches/hyp. Watch where the "
          "metrics plateau:")
    for cycles, npc, lbl in [(1, 6, "~1 ranking round (12 matches)"),
                             (2, 6, "2 rounds (24 matches)"),
                             (4, 6, "4 rounds (48 matches)"),
                             (6, 8, "6 rounds, 8/cycle (96 matches)"),
                             (10, 10, "10 rounds, 10/cycle (200 matches)")]:
        print(_fmt(evaluate(lbl, n=N, cycles=cycles, n_per_cycle=npc,
                            judge_skill=skill)))

    print(f"\n[2] K-FACTOR  (n={N}, fixed 48-match budget, judge_skill={skill:.0%})")
    for lbl, k_fn in [("current 40/20/10", CURRENT_K),
                      ("constant K=32 (FIDE classic)", k_const(32)),
                      ("constant K=16", k_const(16)),
                      ("gentle 24/16/8", k_schedule([(5, 24), (15, 16)], 8)),
                      ("hot 64/32/16", k_schedule([(5, 64), (15, 32)], 16))]:
        print(_fmt(evaluate(lbl, n=N, cycles=4, n_per_cycle=6,
                            judge_skill=skill, k_fn=k_fn)))

    print(f"\n[3] JUDGE QUALITY  (n={N}, 48-match budget) — why a fair judge matters")
    for lbl, skill_v, pbias in [("strong judge (skill 80%)", 0.80, 0.0),
                                ("decent judge (skill 70%)", 0.70, 0.0),
                                ("weak judge (skill 60%)", 0.60, 0.0),
                                ("decent + 15% position bias", 0.70, 0.15)]:
        print(_fmt(evaluate(lbl, n=N, cycles=4, n_per_cycle=6,
                            judge_skill=skill_v, position_bias=pbias)))

    print("\n" + "=" * 100)
    print("Read the takeaways in benchmark/BENCHMARKING_PLAN.md")
    print("=" * 100)


if __name__ == "__main__":
    main()
