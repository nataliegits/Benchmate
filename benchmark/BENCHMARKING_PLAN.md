# Benchmarking Benchmate's Elo tournament

The goal: make the leaderboard *trustworthy* — when Benchmate says hypothesis X is
#1, that should mean X is actually the best, and a re-run should say so again.

Right now there's no way to know if that's true. This plan adds the instrumentation,
and the `benchmark/` package implements it. Below: the four things worth measuring,
what the free simulator already tells us, and the order to do the work.

---

## TL;DR — what to change first

1. **Run more matches.** This is the dominant lever by far. Your saved `state.json`
   has ~2 matches per hypothesis; at that budget the ranking is barely better than
   random (Spearman ≈ 0.20, the true-best lands #1 **0%** of the time). Get to
   ~8–12 matches/hypothesis and Spearman jumps to 0.87–0.93. Concretely: raise
   `n_matches` in `ranking()` and/or run `ranking` on more iterations.
2. **Fix the judge before anything fancy.** Improving judge quality from "decent"
   to "strong" buys as much accuracy as *doubling* the match budget. Use the
   order-swapped `fair_judge` to kill position bias, and give the judge the
   experiment + reviewer notes (it currently only sees statement + rationale).
3. **Don't bother tuning the K-factor.** At your scale (a few dozen matches) the
   40/20/10 schedule, a flat K=16, and a flat K=32 are within noise of each other.
   The chess taper is built for hundreds of games; it's not your problem.

Everything below is evidence for these three claims.

---

## The four pillars

### 1. Trustworthy leaderboard  (`simulate.py`, `metrics.py`) — FREE

The question: *does the Elo ranking converge, and is it stable?*

We can't answer this by running the real loop (costs money, one noisy sample). So
`simulate.py` replays your **actual** `elo.py` (`schedule_matches` + `update_elo`)
against synthetic hypotheses whose true quality we control, thousands of times. It
reports, against a known ground-truth ranking:

| Metric | Meaning | Want |
|---|---|---|
| `spearman` | rank correlation of Elo result vs true order | → 1.0 |
| `top1` | P(true-best hypothesis finishes #1) | high |
| `top3` | P(true-best is in the top 3) | ~1.0 |
| `repeat` | P(two independent runs pick the same #1) | high |
| `churn` | rank movement in the final cycle | → 0 (converged) |

**Findings (n=6 hypotheses, judge skill 70%, 300 replicates each):**

```
MATCH BUDGET
  ~2 matches/hyp (your state.json today)   spearman 0.20   top1  0%   repeat 41%   churn 2.25  ← noise
   4 matches/hyp                           spearman 0.64   top1  5%   repeat 30%   churn 1.35
   8 matches/hyp                           spearman 0.87   top1 65%   repeat 50%   churn 0.39  ← usable
  12 matches/hyp                           spearman 0.93   top1 85%   repeat 74%   churn 0.32
  19 matches/hyp                           spearman 0.97   top1 90%   repeat 83%   churn 0.15  ← solid
```

Read it as: **top-3 is reliable by ~8 matches/hyp, but a trustworthy *#1* needs
~12+.** Since your demo headlines the single top hypothesis, budget for 12.

`repeat` never reaches 100% because there's real signal-limited ambiguity between
near-tied hypotheses — that's honest, not a bug. If two hypotheses are genuinely
close, the system *shouldn't* be certain which is #1.

### 2. Validate vs ground truth  (`gold_set.py`, `run_benchmark.py validate`) — LIVE

The simulator proves the *math* converges. This proves the *real LLM judge* ranks
real hypotheses correctly. `gold_set.py` holds hypotheses for one goal written at
three quality tiers (A > B > C); a trustworthy tournament must recover that order.

```
python -m benchmark.run_benchmark validate --cycles 6 --n-per-cycle 8
```

Reports Spearman between the live Elo leaderboard and the gold tier order, plus
whether a tier-A hypothesis wins. Swap in your own domain's hypotheses — keep the
tier labels honest. This is also your regression test: re-run it after any prompt
or model change and watch the number.

### 3. Compare configurations  (`run_benchmark.py compare`) — LIVE

The question: *does a change actually make the output better, or just different?*
The harness ranks the **same gold set** under two configs and compares each to gold.
Shipped comparison: fair judge vs the current naive judge.

```
python -m benchmark.run_benchmark compare
```

Same pattern extends to the comparisons you care about:
- **Geneformer on vs off** — run the full loop twice (`BENCHMATE_*` env / a flag),
  score each leaderboard against gold. Does your CSV evidence raise the score?
- **Ranking model Haiku vs Sonnet** — `BENCHMATE_MODEL_RANKING=anthropic/claude-sonnet-4-6`
  then `judge-eval`; does accuracy rise enough to justify the cost?

Rule: a config change is only "better" if it raises Spearman-to-gold (or judge
accuracy) beyond the run-to-run noise band — run each config ≥3× and compare ranges.

### 4. Better / fairer judge  (`fair_judge.py`, `judge_eval.py`) — LIVE

The judge is the engine's accuracy ceiling — see how hard judge skill moves the
simulator (strong 0.94 vs weak 0.71 Spearman). Two concrete defects in
`agents.ranking` today:

- **Position bias.** Hypotheses are always shown in a fixed A/B order, and the
  scheduler lists the higher-Elo one first — so any first-slot bias *entrenches the
  current leader* and makes the ranking self-fulfilling.
- **Thin evidence.** The judge sees `statement` + `rationale` only; not the
  `experiment` (is it falsifiable?) or the reviewer notes.

`fair_judge.judge_pair` fixes both: it judges each pair **twice with the order
swapped** and only awards a win when both passes agree on the same hypothesis. If
the verdict flips with order, that's position bias — it's flagged and scored as a
draw instead of letting noise move ratings. The prompt also includes the experiment
and the latest reviewer note.

`judge_eval.py` measures the judge on the gold set:

| Diagnostic | Want |
|---|---|
| accuracy on clear cross-tier pairs | > 80% |
| position-bias rate (verdict flips with order) | < 15% |
| self-consistency (same call twice) | > 85% |
| transitivity violations (A>B>C>A cycles) | ~0 |

```
python -m benchmark.run_benchmark judge-eval --max-pairs 8
```

**Drop-in:** in `co_scientist/graph.py`,
`from benchmark.fair_judge import ranking_fair` and
`g.add_node("ranking", ranking_fair)`. It costs ~2× judge calls per match —
fine, because they're Haiku, and you're spending the savings on correctness.

---

## Recommended order of work

1. **Wire up `validate` and `judge-eval`** (live, small). Get today's baseline
   numbers — you can't improve what you haven't measured.
2. **Switch to `fair_judge`** and re-run `judge-eval`. Confirm position-bias drops.
3. **Raise the match budget** to ~12 matches/hyp (bump `n_matches`, and/or let
   `ranking` run on more iterations). Re-run `validate`.
4. **Re-baseline `compare`** for the configs you care about (Geneformer on/off,
   judge model). Lock in whatever beats gold by more than the noise band.
5. *(Optional, later)* If you scale to many more hypotheses, revisit the rating
   system itself — Glicko-2 adds a rating-deviation term (uncertainty per
   hypothesis), which is more sample-efficient than plain Elo at small match
   counts. Not worth it yet.

## Notes / caveats on the simulator

- Match outcomes use a Bradley–Terry model with an interpretable `judge_skill`
  knob (P(judge orders an adjacent pair correctly)). Real judge skill is unknown —
  that's exactly what `judge-eval` measures, so feed the measured accuracy back in
  as `judge_skill` for a calibrated simulation.
- The position-bias row in the simulator understates real harm: because the
  scheduler lists the leader first, simulated bias mostly reinforces an
  already-correct order. The *real* danger (freezing an early mistake) is better
  caught by `judge-eval`'s direct flip-rate measurement.
- All free results are 300 replicates; rerun `simulate.py` to regenerate.
