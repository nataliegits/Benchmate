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
| `results.py` | Saves each run to `results/*.json` so the hosted app can display numbers it didn't compute |

The `--ontology` flag injects canonical ontology terms (via OntoMCP) into the
fair judge's prompt — the structured-knowledge layer in `co_scientist/ontology.py`.
Start the server first: `git clone https://github.com/jeanlouishoneine-tech/OntoMCP
&& cd OntoMCP && make install && make serve-api`, then set `ONTOMCP_API_URL` if it
isn't on `http://localhost:8000`. Grounding is fail-soft: with the server off, the
flag exits with setup instructions and the other commands are unaffected.

## Running the benchmarks yourself (anyone, with your own key)

You don't have to use the CLI — the **Benchmark tab** in the Streamlit app runs
all of these with sliders and buttons, and saves each run so it shows up later
(and on the hosted demo).

**In the hosted app:**

1. Open the app and put **your own** Anthropic key in the sidebar. You pay only
   for your own calls; the maintainer's key is never used.
2. Go to the **Benchmark** tab. The free **simulator** runs with no key. The
   **judge accuracy**, **validate**, and **fair-vs-naive** buttons unlock once
   your key is set.
3. Each section shows the maintainer's last saved result by default; click the
   button to run your own on top of it.

**The ontology comparison is the exception.** It needs a local OntoMCP server,
which isn't part of the hosted site. To run that one yourself:

```bash
git clone https://github.com/jeanlouishoneine-tech/OntoMCP.git && cd OntoMCP
uv sync && uv run ontomcp-api          # leave running; serves http://localhost:8000

# in another terminal, with the Benchmate app running locally + your key set:
#   the "Run ontology compare" button turns on once OntoMCP is reachable
```

Prefer the terminal? Every button has a CLI twin: `python -m benchmark.run_benchmark
<simulate|judge-eval|validate|compare|compare --ontology>` (see Commands above).

## The one change to try first

In `co_scientist/graph.py`:

```python
from benchmark.fair_judge import ranking_fair
g.add_node("ranking", ranking_fair)   # was: agents.ranking
```

…then raise the match budget (see the plan). Re-run `validate` before and after.
