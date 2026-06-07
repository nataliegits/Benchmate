# Benchmate

A small AI Co-Scientist for biomedical hypothesis generation. Seven LangGraph
agents (Generation, Reflection, Ranking, Proximity, Evolution, Meta-review,
Supervisor) talk to each other in a loop, propose hypotheses, run a pairwise
Elo tournament, and refine the winners. PubMed is wired in as a real tool.
**Geneformer in-silico perturbation results are wired in as cached evidence**
so the agents can reason about your own experimental data, not just the
literature.

Try it: **[benchmate.streamlit.app](https://benchmate.streamlit.app)**

## What's in here

```
Benchmate/
├── README.md
├── requirements.txt
├── .env.example
├── run.py                  # CLI entry: python run.py "your research goal"
├── co_scientist/
│   ├── state.py            # shared state: hypotheses, Elo, memory
│   ├── elo.py              # Elo math + tournament scheduler
│   ├── tools.py            # PubMed + Geneformer lookup + BioNeMo stub
│   ├── llm.py              # litellm wrapper, multi-provider routing
│   ├── llm_config.py       # per-role model assignments
│   ├── agents.py           # the 7 agents (Generation reads Geneformer cache,
│   │                       # Reflection fact-checks against it)
│   └── graph.py            # LangGraph wiring of the supervisor loop
├── benchmark/              # Is the Elo leaderboard trustworthy? (see below)
│   ├── simulate.py         # free Monte Carlo over the real elo.py
│   ├── metrics.py          # Spearman, top-k, churn, transitivity
│   ├── fair_judge.py       # order-swapped, bias-aware judge (drop-in)
│   ├── judge_eval.py       # live judge accuracy / position-bias / consistency
│   ├── gold_set.py         # tier A/B/C gold hypotheses for validation
│   ├── run_benchmark.py    # CLI: simulate | judge-eval | validate | compare
│   └── BENCHMARKING_PLAN.md
├── notebooks/              # Geneformer perturbation notebooks (Colab)
│   ├── 01_geneformer_erad_perturbation.ipynb
│   └── 02_geneformer_ciliated_cells.ipynb
├── data/geneformer/        # cached perturbation results (CSV, gitignored)
│   └── README.md           # expected CSV schema
├── ui/                     # Streamlit UI
│   ├── app.py              # 4-tab Streamlit app
│   ├── notebook_gen.py     # parameterise notebook 02 with user's genes
│   ├── colab_handoff.py    # push notebook to Gist (API or gh CLI)
│   └── watcher.py          # optional: Drive sync folder watcher
├── hermes/                 # Hermes Agent integration
│   └── benchmate_runner.py # JSON API for chat-driven Benchmate
├── HERMES.md               # Hermes VPS deployment guide
└── DEPLOY.md               # Streamlit Cloud deployment guide
```

## Two ways to use it

### A. CLI

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set ANTHROPIC_API_KEY

python run.py "Find novel drug repurposing candidates for acute myeloid leukemia"
```

Iterations print top hypotheses and Elo ratings as they go. State is
checkpointed to `state.json`; resume with `python run.py --resume`.

### B. Streamlit UI

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`. Four tabs:

1. **New perturbation.** Type gene symbols, pick a cell context. Resolves
   to Ensembl IDs via mygene, generates a parameterised copy of
   `notebooks/02_geneformer_ciliated_cells.ipynb` with your genes pre-filled,
   pushes it to a GitHub Gist, returns a one-click "Open in Colab" link.
2. **Inspect cache.** Browse what's in `data/geneformer/` — pick a gene,
   see the top-N affected, sortable.
3. **Run Benchmate.** Paste a research goal, choose iterations, run. Logs
   stream into the page. State downloads when finished.
4. **Hermes preview.** See the JSON shape Hermes receives when wired up
   for chat-driven runs.

## The Geneformer integration

When the research goal (or a hypothesis under review) mentions a gene symbol
that has a cached `{GENE}_stats.csv` in `data/geneformer/`, the **Generation**
agent injects that gene's top-10 affected genes into its prompt as evidence,
and the **Reflection** agent uses the same evidence to fact-check proposed
mechanisms. The lookup is a pandas read; latency is milliseconds.

To populate the cache:

- Run `notebooks/02_geneformer_ciliated_cells.ipynb` in Colab with your
  chosen `TARGETS`, or use the UI's tab 1 to generate the notebook for you.
- Drop the resulting `*_stats.csv` files into `data/geneformer/` (manually,
  via the UI's Upload CSVs sidebar, or via `ui/watcher.py`).

See `data/geneformer/README.md` for the expected CSV schema.

## Benchmarking the Elo tournament

Benchmarks are built in. Benchmate ships its own toolkit for the question
its headline number depends on — when the leaderboard says hypothesis X is
#1, is X actually the best, and would a re-run agree? Same four entry
points are available **two ways**, no separate install:

- **Streamlit app, *Benchmark* tab.** The free simulator runs in-browser
  in seconds (no API key). The live benchmarks have their own buttons,
  show an upper-bound cost estimate, and read your API key from the
  sidebar.
- **CLI**, for scripting and regression runs:
  ```bash
  python -m benchmark.run_benchmark simulate     # FREE — Monte Carlo over the real elo.py
  python -m benchmark.run_benchmark judge-eval   # LIVE — is the real LLM judge any good?
  python -m benchmark.run_benchmark validate     # LIVE — rank the gold set, score vs tier order
  python -m benchmark.run_benchmark compare      # LIVE — fair vs naive judge, side by side
  ```

The gold set in `benchmark/gold_set.py` is ERAD / bortezomib-resistant
multiple myeloma (tiered A > B > C); swap it for your own domain.

### How to actually prove the ranking is accurate

Benchmate follows a six-step protocol from "is the math right?" to "does
the real loop pick the right hypothesis?" — math sanity → match-budget
sweep → live judge accuracy → end-to-end validation → reproducibility →
robustness to the known failure mode. Full protocol with pass bars and
which tool to run at each step is in
[`benchmark/BENCHMARKING_PLAN.md`](benchmark/BENCHMARKING_PLAN.md).

### What the free simulator already tells you

The dominant lever is match count, not K-factor. At ~2 matches/hypothesis
(default state) Spearman vs ground truth is ~0.20 and the true best lands
#1 about 0% of the time; at ~12 matches/hypothesis Spearman is 0.93 and
#1 is correct 85% of the time. The 40/20/10 K-schedule, flat K=16, and
flat K=32 are within noise of each other at this scale.

`benchmark/fair_judge.ranking_fair` is the default ranking node in
`co_scientist/graph.py` — it judges each pair twice with the order
swapped and scores a draw when the verdict flips. Pass `--naive-judge`
to `run.py` to fall back. The per-round match budget is exposed as
`--n-matches` (default 8); aim for ~12 matches per hypothesis across
the whole run.

## Multi-model routing

Each agent role has its own LLM assignment via litellm. The default sends
Generation, Reflection, and Evolution to Claude Sonnet 4.6 (where the
Geneformer evidence has to be reasoned about) and Ranking, Meta-review,
and Supervisor to Haiku 4.5 (where throughput matters). Override per role
in the UI sidebar's *Model routing* panel, or set
`BENCHMATE_MODEL_<ROLE>=provider/model` as an env var.

Switching the throughput roles from Sonnet to Haiku cuts an 8-iteration
loop from ~$2.50 to ~$0.90 with no measurable hypothesis-quality loss.
The same role-routing pattern works with any model litellm supports —
Anthropic, OpenAI, Google, Mistral, Bedrock, and more.

## Hermes integration

`hermes/benchmate_runner.py` exposes Benchmate's main operations as
JSON-in / JSON-out functions that the [Hermes Agent](https://hermes-agent.nousresearch.com/)
can call via skills. Four functions: `list_cache`, `gene_neighbors`,
`add_perturbation`, `run_benchmate`. Each is also runnable as a CLI:

```bash
python -m hermes.benchmate_runner list-cache
python -m hermes.benchmate_runner neighbors TXNDC15 --top-n 10
python -m hermes.benchmate_runner add-perturbation FOXP3 \
    --cell-context "Plasma cells (extreme ER stress)"
python -m hermes.benchmate_runner run "your research goal" --max-iterations 8
```

Wire Benchmate behind Hermes on a small VPS and you can trigger runs from
Slack, Discord, or Telegram, or schedule them via natural-language cron.
Full deployment guide in `HERMES.md`.

## How to extend

Productive sequence, roughly in order of value:

1. **Run it once end-to-end** — both the CLI and the UI — before changing
   anything. Get a feel for what the agents produce.
2. **Add more genes to the cache.** Every new gene you perturb makes the
   Generation agent strictly more useful for goals that mention it.
3. **Wire your own evidence sources** in `co_scientist/tools.py` — your
   RNA-seq differential expression, your PPI data, your DepMap viability
   scores. The pattern from `geneformer_neighbors` and
   `_geneformer_context_for` generalises directly.
4. **Tune the per-role model routing** in `co_scientist/llm_config.py` or
   via the UI sidebar to your cost/quality preference.
5. **Add LangSmith tracing** — set `LANGSMITH_API_KEY` and LangGraph
   traces automatically.

## On honesty

This is a small skeleton of Google's Co-Scientist architecture, not a
replica of its quality. Google ran their version for many hours per
research goal with Gemini 2.5 Pro and many tools. Benchmate's value isn't
in matching that — it's in giving you a loop you can fully reason about,
then extending it with the specific evidence sources your specific
research actually needs.
