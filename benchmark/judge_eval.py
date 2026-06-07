"""Measure whether the LLM judge is any good. LIVE — uses real API calls.

Four diagnostics, all on the gold set so we know the right answers:

  1. accuracy          — on cross-tier pairs (clear winner), how often does the
                         judge pick the better hypothesis?
  2. position_bias     — how often does the verdict flip when we swap A/B order?
                         (this is the bias the fair judge neutralises)
  3. self_consistency  — same pair, same order, asked twice: agreement rate.
  4. transitivity      — among the gold set, how many intransitive triples
                         (A>B, B>C, C>A)?

Cost is bounded by --max-pairs. Defaults are small. Set ANTHROPIC_API_KEY first.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from co_scientist.llm import call_json
from .fair_judge import judge_pair, _ask
from .gold_set import GOLD, gold_hypotheses, gold_pairs
from .metrics import transitivity_violations

CRITERIA = ["novelty", "mechanistic plausibility", "falsifiability of the experiment"]


@dataclass
class JudgeReport:
    n_pairs: int
    accuracy: float
    position_bias_rate: float
    self_consistency: float
    transitivity_violations: int
    transitivity_triples: int


def _naive_winner(a, b, role="ranking") -> str:
    """The CURRENT judge: single pass, fixed order, statement+rationale only.
    Returns 'A'/'B'/'draw'. Used to measure position bias of the status quo."""
    out = call_json(
        f"Two competing hypotheses. Judge which is better.\n\n"
        f"A: {a.statement}\n   rationale: {a.rationale}\n\n"
        f"B: {b.statement}\n   rationale: {b.rationale}\n\n"
        f"Evaluation criteria: {', '.join(CRITERIA)}\n\n"
        'Output JSON: {"winner": "A" | "B" | "draw", "reason": "..."}',
        system="You are a rigorous, impartial scientific reviewer.",
        role=role, temperature=0.2, max_tokens=300,
    )
    return str(out.get("winner", "draw")).strip().upper()[:1].replace("D", "draw") or "draw"


def evaluate_judge(max_pairs: int = 8, role: str = "ranking") -> JudgeReport:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the live judge eval.")

    hyps = gold_hypotheses()
    pairs = gold_pairs()[:max_pairs]

    correct = bias = consistent = 0
    for i, j, winner_idx in pairs:
        a, b = hyps[i], hyps[j]
        # accuracy + position bias from the two order-swapped passes
        v = judge_pair(a, b, CRITERIA, role=role)
        # the gold winner is index i, which is hypothesis "A" here
        if v.winner == "A":
            correct += 1
        bias += int(v.position_biased)
        # self-consistency: same order, asked again
        r_again, _ = _ask(a, b, ", ".join(CRITERIA), "", role, None)
        again = {"first": "A", "second": "B", "draw": "draw"}[r_again]
        first_pass = "A" if v.reasons[0] and v.winner != "draw" else v.winner
        consistent += int(again == first_pass)

    n = len(pairs)

    # transitivity over the full gold set (one judgment per unordered pair)
    beats: dict[tuple[int, int], int] = {}
    for i in range(len(hyps)):
        for j in range(i + 1, len(hyps)):
            v = judge_pair(hyps[i], hyps[j], CRITERIA, role=role)
            if v.winner == "A":
                beats[(i, j)] = i
            elif v.winner == "B":
                beats[(i, j)] = j
            # draws omitted from transitivity check
    viol, triples = transitivity_violations(beats)

    return JudgeReport(
        n_pairs=n,
        accuracy=correct / n if n else float("nan"),
        position_bias_rate=bias / n if n else float("nan"),
        self_consistency=consistent / n if n else float("nan"),
        transitivity_violations=viol,
        transitivity_triples=triples,
    )


def print_report(r: JudgeReport, role: str) -> None:
    print(f"\nJUDGE SCORECARD  (model role='{role}', {r.n_pairs} cross-tier pairs)")
    print(f"  accuracy on clear pairs : {r.accuracy:5.0%}   "
          "(picks the better hypothesis; want > 80%)")
    print(f"  position-bias rate      : {r.position_bias_rate:5.0%}   "
          "(verdict flips with A/B order; want < 15%)")
    print(f"  self-consistency        : {r.self_consistency:5.0%}   "
          "(same call twice agrees; want > 85%)")
    tv = (f"{r.transitivity_violations}/{r.transitivity_triples}"
          if r.transitivity_triples else "n/a")
    print(f"  transitivity violations : {tv}   (A>B>C>A cycles; want ~0)")
    if r.accuracy < 0.8 or r.position_bias_rate > 0.15:
        print("  --> Judge is the bottleneck. Use fair_judge + a stronger "
              "ranking model before adding matches.")
