"""Demo: walk a set of 6 hypotheses through the evaluative agents.

The same six hypotheses (the gold set) pass through three LLM-as-judge roles:

  1. REFLECTION  — the critic.   Reviews each hypothesis on its own merits.
  2. RANKING     — the referee.  Pairwise Elo tournament; this is where the
                                 leaderboard is built.
  3. META-REVIEW — the editor.   Reads the top vs bottom of the board and writes
                                 a lesson that feeds back into the next round.

By default this runs **free** with deterministic stand-in agents (no API), so
it's safe to demo live. Pass --live to use the real Benchmate agents + judge.

    python -m benchmark.demo_walkthrough                 # free, deterministic
    python -m benchmark.demo_walkthrough --skill 0.75    # noisier referee
    python -m benchmark.demo_walkthrough --live          # real API calls
"""
from __future__ import annotations

import argparse
import random

import co_scientist.elo as elo
from co_scientist.state import Hypothesis, INITIAL_ELO
from benchmark.gold_set import GOLD
from benchmark.metrics import spearman

CRITERIA = ["novelty", "mechanistic plausibility", "falsifiability of the experiment"]

# Canned critiques for the free/deterministic mode, keyed by gold tier.
_MOCK_CRITIQUE = {
    "A": "Names a specific molecular mechanism and proposes a falsifiable "
         "experiment WITH a control — strong and testable.",
    "B": "Plausible, but the mechanism is unnamed and the experiment lacks a "
         "control — hard to falsify cleanly.",
    "C": "Circular: restates the phenotype as its own cause. The experiment "
         "can't distinguish the hypothesis from the null.",
}


def build_hypotheses() -> list[Hypothesis]:
    hyps = []
    for i, g in enumerate(GOLD):
        h = Hypothesis.new(statement=g["statement"], rationale=g["rationale"],
                           experiment=g["experiment"])
        h.elo = float(INITIAL_ELO)
        h.matches_played = 0
        h.meta = {"label": f"H{i}", "tier": g["tier"], "gold": i}  # demo bookkeeping
        hyps.append(h)
    return hyps


def lbl(h): return h.meta["label"]
def tier(h): return h.meta["tier"]


# --------------------------------------------------------------------------
# STAGE 1 — REFLECTION (the critic)
# --------------------------------------------------------------------------

def stage_reflection(hyps, live):
    print("\n" + "=" * 72)
    print("STAGE 1 · REFLECTION  (the critic) — reviews each hypothesis alone")
    print("=" * 72)
    if live:
        from co_scientist.agents import reflection
        state = {"hypotheses": hyps, "plan_config": {"evaluation_criteria": CRITERIA}}
        reflection(state)                       # mutates h.review_notes in place
        for h in hyps:
            note = h.review_notes[-1] if h.review_notes else "(not reviewed this pass)"
            print(f"\n{lbl(h)} ({tier(h)}): {note}")
    else:
        for h in hyps:
            note = _MOCK_CRITIQUE[tier(h)]
            h.review_notes.append(note)
            print(f"\n{lbl(h)} ({tier(h)}): {note}")


# --------------------------------------------------------------------------
# STAGE 2 — RANKING (the referee)
# --------------------------------------------------------------------------

def _mock_judge(a, b, skill: float, rng: random.Random):
    """Deterministic-ish referee for free mode. Picks the better gold tier with
    probability `skill`; same-tier pairs are near coin-flips."""
    ga, gb = a.meta["gold"], b.meta["gold"]
    better, worse = (a, b) if ga < gb else (b, a)
    same_tier = a.meta["tier"] == b.meta["tier"]
    p = 0.55 if same_tier else skill
    return better if rng.random() < p else worse


def stage_ranking(hyps, rounds, n_per_cycle, skill, seed, live):
    print("\n" + "=" * 72)
    print("STAGE 2 · RANKING  (the referee) — pairwise Elo tournament")
    print("=" * 72)
    rng = random.Random(seed)
    random.seed(seed)                            # schedule_matches uses global random

    if live:
        from benchmark.fair_judge import judge_pair
        def decide(a, b):
            v = judge_pair(a, b, CRITERIA)
            return a if v.winner == "A" else (b if v.winner == "B" else None)
    else:
        def decide(a, b):
            return _mock_judge(a, b, skill, rng)

    total = 0
    for r in range(1, rounds + 1):
        matches = elo.schedule_matches(hyps, n_matches=n_per_cycle)
        print(f"\n--- round {r}: {len(matches)} matches ---")
        for a, b in matches:
            w = decide(a, b)
            if w is None:                        # draw (live judge flagged a tie/bias)
                elo.update_elo(a, b, draw=True)
                print(f"  {lbl(a)}({tier(a)}) vs {lbl(b)}({tier(b)})  ->  draw")
            else:
                elo.update_elo(w, b if w is a else a)
                print(f"  {lbl(a)}({tier(a)}) vs {lbl(b)}({tier(b)})  ->  {lbl(w)} wins")
            total += 1
        board = sorted(hyps, key=lambda h: -h.elo)
        print("  standings: " + "  ".join(f"{lbl(h)}({tier(h)}) {h.elo:.0f}" for h in board))
    return total


# --------------------------------------------------------------------------
# STAGE 3 — META-REVIEW (the editor)
# --------------------------------------------------------------------------

def stage_meta_review(hyps, live):
    print("\n" + "=" * 72)
    print("STAGE 3 · META-REVIEW  (the editor) — patterns from top vs bottom")
    print("=" * 72)
    board = sorted(hyps, key=lambda h: -h.elo)
    top, bottom = board[:2], board[-2:]
    print("  top 2:    " + ", ".join(f"{lbl(h)}({tier(h)})" for h in top))
    print("  bottom 2: " + ", ".join(f"{lbl(h)}({tier(h)})" for h in bottom))
    if live:
        from co_scientist.agents import meta_review
        out = meta_review({"hypotheses": hyps})
        crit = out.get("meta_critique")
        if crit:
            for x in crit.recurring_issues:
                print(f"   - avoid: {x}")
            for x in crit.successful_patterns:
                print(f"   - keep:  {x}")
    else:
        print("   - keep:  name a specific mechanism and include a control")
        print("   - avoid: restating the phenotype as its own explanation")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-per-cycle", type=int, default=8)
    ap.add_argument("--skill", type=float, default=0.85,
                    help="free-mode referee skill on cross-tier pairs (0.5-1.0)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live", action="store_true", help="use real agents + API")
    args = ap.parse_args()

    hyps = build_hypotheses()
    print("SIX HYPOTHESES (all start at 1200):")
    for h in hyps:
        print(f"  {lbl(h)} · tier {tier(h)} · {h.statement[:70]}...")

    stage_reflection(hyps, args.live)
    total = stage_ranking(hyps, args.rounds, args.n_per_cycle, args.skill, args.seed, args.live)
    stage_meta_review(hyps, args.live)

    board = sorted(hyps, key=lambda h: -h.elo)
    gold_scores = [-h.meta["gold"] for h in hyps]      # higher = better
    rho = spearman(gold_scores, [h.elo for h in hyps])
    print("\n" + "=" * 72)
    print(f"RESULT  ({total} matches, ~{total*2/len(hyps):.1f} per hypothesis)")
    print("  final:  " + " > ".join(f"{lbl(h)}({tier(h)})" for h in board))
    print("  gold:   " + " > ".join(f"H{i}({GOLD[i]['tier']})" for i in range(len(GOLD))))
    print(f"  spearman vs gold: {rho:+.2f}   (1.0 = perfect tier recovery)")
    print("=" * 72)


if __name__ == "__main__":
    main()
