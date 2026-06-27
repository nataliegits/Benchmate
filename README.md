# Benchmate

A small AI Co-Scientist for biomedical hypothesis generation. Seven LangGraph
agents (Generation, Reflection, Ranking, Proximity, Evolution, Meta-review,
Supervisor) talk to each other in a loop, propose hypotheses, run a pairwise
Elo tournament, and refine the winners. PubMed is wired in as a real tool.
**Geneformer in-silico perturbation results are wired in as cached evidence**
so the agents can reason about your own experimental data, not just the
literature.

On top of that, Benchmate adds two evaluation layers beyond the LLM judge: a
**structured-knowledge layer** (OntoMCP) that grounds the judge in canonical
biology, and a **"cross-check with other models" panel** that scores hypotheses
with independent quantitative models — **AlphaGenome** (regulatory effect from
DNA), **Boltz** (structure & binding), **Open Targets** (gene↔disease
association), **DepMap** (gene dependency), and **AlphaMissense** (variant
pathogenicity). The idea: a trustworthy co-scientist needs a *panel of judges* —
semantic, structured, and quantitative — each catching what the others miss.

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
│   ├── graph.py            # LangGraph wiring of the supervisor loop
│   ├── ontology.py         # OntoMCP grounding: canonical ontology terms fed to
│   │                       # the judge + Generation/Reflection/Proximity/Supervisor
│   ├── variant_scorer.py   # AlphaGenome / Enformer — regulatory-effect scoring
│   ├── boltz_scorer.py     # Boltz API — structure & binding scoring
│   └── target_scorer.py    # Open Targets + DepMap + AlphaMissense scorers
├── benchmark/              # Is the leaderboard trustworthy + how does it cross-check?
│   ├── simulate.py         # free Monte Carlo over the real elo.py
│   ├── metrics.py          # Spearman, top-k, churn, transitivity
│   ├── fair_judge.py       # order-swapped, bias-aware judge (drop-in)
│   ├── judge_eval.py       # live judge accuracy / position-bias / consistency
│   ├── gold_set.py         # tier A/B/C gold hypotheses for validation
│   ├── gold_set_adversarial.py   # solid / fluent-but-false / novel-but-true
│   ├── run_adversarial.py        # ontology discrimination test
│   ├── alias_dedup.py            # ontology vs text similarity (identity)
│   ├── gold_set_variants.py · fetch_eqtls.py · build_variant_scores.py  # AlphaGenome
│   ├── gold_set_binding.py · fetch_uniprot.py · build_boltz_scores.py   # Boltz binding cross-check
│   ├── gold_set_genes.py · build_target_scores.py    # Open Targets / DepMap gene cross-check
│   ├── elo_vs_variant_score.py   # Elo vs an independent quantitative score
│   ├── results.py          # saves runs so the hosted app can display them
│   ├── run_benchmark.py    # CLI: simulate | judge-eval | validate | compare [--ontology]
│   └── BENCHMARKING_PLAN.md
├── notebooks/              # Geneformer perturbation notebooks (Colab)
│   ├── 01_geneformer_erad_perturbation.ipynb
│   └── 02_geneformer_ciliated_cells.ipynb
├── data/geneformer/        # cached perturbation results (CSV, gitignored)
│   └── README.md           # expected CSV schema
├── ui/                     # Streamlit UI
│   ├── app.py              # 6-tab Streamlit app
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

Opens at `http://localhost:8501`. Six tabs:

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
5. **Benchmark.** Is the Elo leaderboard trustworthy? The free simulator,
   the judge diagnostics, validate-vs-gold, fair-vs-naive, the ontology
   grounding compare, and the discrimination + alias-dedup tests.
6. **Cross-check with other models.** Correlate the Elo ranking against
   independent quantitative models — AlphaGenome (regulatory), Boltz
   (binding), Open Targets (association), DepMap (dependency), and
   AlphaMissense (pathogenicity). Low correlation flags hypotheses the
   leaderboard alone shouldn't pick for the bench.

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

## Ontology grounding for the judge

`co_scientist/ontology.py` is the **structured-knowledge layer** — the same
evidence-injection pattern as Geneformer. It started on the Ranking judge and is
now fed to five agents (judge, Generation, Reflection, Proximity, Supervisor).
Before the fair judge compares two hypotheses, it resolves the biological
entities each one mentions (genes, proteins, complexes, diseases, small
molecules, cell types) to canonical ontology terms and appends them to the
judge prompt as a *"known-biology grounding"* block.

It talks to [**OntoMCP**](https://github.com/jeanlouishoneine-tech/OntoMCP), a
small MCP server + HTTP API over EBI's OLS4 that returns real CURIEs with no
hallucinated IDs. Three properties matter:

- **Background, not veto.** The injected text tells the judge these are
  background facts and that *absence of a term is not evidence against a
  hypothesis* — the "don't hard-veto novel biology" caveat, baked in.
- **Fail-soft everywhere.** Server down, slow, or returning an unexpected JSON
  shape → empty grounding, never a crash. A tournament runs fine with OntoMCP
  off; grounding is a pure enhancement.
- **Defensive parser.** `_iter_hits` / `_norm_hit` accept several plausible
  OLS/OntoMCP response shapes and extract CURIEs, labels, and synonyms (e.g.
  `HRD1 → aka SYVN1`), shrugging off shapes they don't recognise.

Start OntoMCP, then point Benchmate at it:

```bash
git clone https://github.com/jeanlouishoneine-tech/OntoMCP && cd OntoMCP
make install && make serve-api          # -> http://localhost:8000
export ONTOMCP_API_URL=http://localhost:8000   # default already matches
```

Smoke-test the grounding (and see what resolves):

```bash
python -m co_scientist.ontology
```

To measure whether grounding actually helps the ranking, `compare --ontology`
runs the fair judge over the gold set with grounding **OFF vs ON** and prints
the Δ spearman (ontology − baseline). If OntoMCP isn't running it exits with
copy-paste setup instructions rather than erroring mid-run:

```bash
python -m benchmark.run_benchmark compare --ontology
```

(One run is a single noisy sample — run it ≥3× each way and compare ranges
before calling the layer a win.)

## Cross-check with other models (the panel of judges)

The Elo leaderboard is the LLM judge's opinion. Before trusting it to choose a
wet-lab candidate, Benchmate cross-checks it against **independent quantitative
models**, each scoring the slice of hypotheses it can actually speak to. Low
correlation with Elo is a flag. This lives in the **"Cross-check with other
models"** Streamlit tab and in `benchmark/elo_vs_variant_score.py`.

| Model | Question it answers | Tooling | Access |
|---|---|---|---|
| **AlphaGenome** | does a *variant* change expression? | `co_scientist/variant_scorer.py` | free key, score in Colab (needs Py 3.10+) |
| **Boltz** | does a *molecule* bind the target? | `co_scientist/boltz_scorer.py` | plain API, $100 launch credits (`BOLTZLAUNCH`) |
| **Open Targets** | is the *gene* linked to the disease? | `co_scientist/target_scorer.py` | free GraphQL, no key |
| **DepMap** | is the *gene* a real dependency? | `co_scientist/target_scorer.py` | public `CRISPRGeneEffect.csv` |
| **AlphaMissense** | is a *coding variant* pathogenic? | `co_scientist/target_scorer.py` | free, via Ensembl VEP |

Each follows the same recipe: frame the hypotheses the model can score (regulatory
variants for AlphaGenome, protein+ligand pairs for Boltz, genes for Open Targets /
DepMap, coding variants for AlphaMissense), score them, and correlate against the
Elo ranking.

```bash
# AlphaGenome (regulatory):
python -m benchmark.fetch_eqtls            # real GTEx eQTL coordinates
#   ...score in benchmark/alphagenome_scoring_colab.ipynb, then:
python -m benchmark.build_variant_scores   # merge Elo + score -> variant_scores.json

# Boltz (binding):
export BOLTZ_API_KEY=...                    # api.boltz.bio (redeem BOLTZLAUNCH)
python -m benchmark.fetch_uniprot           # real UniProt sequences for the gold set
python -m benchmark.build_boltz_scores      # score + merge -> boltz_scores.json

# Open Targets (free) + DepMap (needs data/depmap/CRISPRGeneEffect.csv):
python -m benchmark.build_target_scores     # -> opentargets_scores.json, depmap_scores.json

# see any correlation:
python -m benchmark.elo_vs_variant_score --scores benchmark/opentargets_scores.json
```

The gold sets (`gold_set_variants.py`, `gold_set_binding.py`, `gold_set_genes.py`)
ship with a deliberate negative control and **placeholder coordinates** where real
data isn't fetched automatically — `fetch_uniprot.py` pulls real protein sequences,
and Open Targets resolves gene/disease IDs by name at query time. Every scorer is
fail-soft: no key / no data → no score, never a crash.

### What the panel found (ERAD / bortezomib-resistant myeloma gold set)

Three independent judges, each scoring a different axis, all **disagree** with the
LLM Elo ranking — the point of having a panel:

| Judge | Axis | Spearman(Elo, model) |
|---|---|---|
| **AlphaGenome** | regulatory effect | **−0.60** |
| **Boltz** | binding confidence | **+0.26** |
| **Open Targets** | disease association | **−0.06** |

The Open Targets result is the sharpest: a near-perfect inversion. The LLM judge
tops the *elaborate* ERAD genes (SYVN1/HRD1, SEL1L, EDEM1); Open Targets' real-world
evidence backs the *proven* drug target (PSMB5, the bottom of the Elo list). The
negative control (OR2T1, an unrelated olfactory receptor) sits last in both — the
sanity check that the disagreement is signal, not noise. n is small (6 per set), so
these are flags to investigate, not verdicts — exactly how a panel of judges should
be read.

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
  python -m benchmark.run_benchmark compare --ontology  # LIVE — fair judge OFF vs ON ontology grounding
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
