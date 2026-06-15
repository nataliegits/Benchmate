# benchmark/ — Benchmate benchmarking toolkit

Tools to measure (and improve) whether the Elo tournament produces a leaderboard
you can trust. Read **BENCHMARKING_PLAN.md** for the strategy and findings; this
is the quick-start.

## Commands

```bash
# FREE — no API key. Monte Carlo over your real elo.py.
# Answers: how many matches / what K-factor give a stable, accurate ranking?
python -m benchmark.run_benchmark simulate

# LIVE — needs ANTHROPIC_API_KEY. Is the LLM judge any good?
# Accuracy, position-bias rate, self-consistency, transitivity on the gold set.
python -m benchmark.run_benchmark judge-eval --max-pairs 8

# LIVE — rank the gold set and score it against the known-correct tier order.
python -m benchmark.run_benchmark validate --cycles 6 --n-per-cycle 8

# LIVE — same gold set, fair judge vs the current naive judge, side by side.
python -m benchmark.run_benchmark compare

# LIVE — same gold set, fair judge with ontology grounding OFF vs ON.
# Prints Δ spearman (ontology − baseline). Needs OntoMCP running (see below).
python -m benchmark.run_benchmark compare --ontology
```

## Files

| File | What it is |
|---|---|
| `simulate.py` | Free Monte Carlo simulator; replays real `elo.py` against synthetic hypotheses |
| `metrics.py` | Pure-Python ranking metrics: Spearman, Kendall, top-k overlap, churn, transitivity |
| `fair_judge.py` | Order-swapped, bias-aware pairwise judge. `ranking_fair` is a drop-in for `agents.ranking` |
| `judge_eval.py` | Live judge diagnostics on the gold set |
| `gold_set.py` | Tiered gold-standard hypotheses for validation — **edit for your domain** |
| `run_benchmark.py` | CLI tying it together. `compare --ontology` toggles the OntoMCP grounding layer |

The `--ontology` flag injects canonical ontology terms (via OntoMCP) into the
fair judge's prompt — the structured-knowledge layer in `co_scientist/ontology.py`.
Start the server first: `git clone https://github.com/jeanlouishoneine-tech/OntoMCP
&& cd OntoMCP && make install && make serve-api`, then set `ONTOMCP_API_URL` if it
isn't on `http://localhost:8000`. Grounding is fail-soft: with the server off, the
flag exits with setup instructions and the other commands are unaffected.

## The one change to try first

In `co_scientist/graph.py`:

```python
from benchmark.fair_judge import ranking_fair
g.add_node("ranking", ranking_fair)   # was: agents.ranking
```

…then raise the match budget (see the plan). Re-run `validate` before and after.
