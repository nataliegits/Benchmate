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

# LIVE — ontology DISCRIMINATION test on the adversarial set:
# does grounding sink the fluent-but-FALSE traps WITHOUT punishing NOVEL ideas?
python -m benchmark.run_adversarial --cycles 6 --n-per-cycle 8

# Does the Elo ranking agree with an independent sequence model (AlphaGenome /
# Enformer)? Build a scores file first (see elo_vs_variant_score.py), or:
python -m benchmark.elo_vs_variant_score --demo
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
| `gold_set_adversarial.py` | 3-kind set (solid / fluent-but-false / novel-but-true) for the discrimination test — **review the biology** |
| `run_adversarial.py` | Ontology discrimination test: reports trap-demotion vs novelty-penalty |
| `elo_vs_variant_score.py` | Correlates the Elo ranking with an independent quantitative score (generic) |
| `gold_set_variants.py` · `fetch_eqtls.py` · `build_variant_scores.py` | AlphaGenome cross-check: variant hypotheses, real GTEx eQTLs, and the Elo+score merge |
| `gold_set_binding.py` · `fetch_uniprot.py` · `build_boltz_scores.py` | **Boltz cross-check**: protein+ligand binding hypotheses, real UniProt sequences, and the Elo+score merge |
| `gold_set_genes.py` · `build_target_scores.py` | **Open Targets / DepMap cross-check**: gene hypotheses and the Elo+score merge |
| `fetch_clinvar.py` · `gold_set_missense.py` · `build_missense_scores.py` | **AlphaMissense cross-check**: real ClinVar missense variants, framed as hypotheses, and the Elo+score merge |

## Cross-check with other models (the "panel of judges")

The Elo leaderboard is the LLM judge's opinion. **Five independent quantitative**
models cross-check it from different angles, each scoring the slice of hypotheses
it can actually speak to. Low correlation with Elo = a flag before the bench.
They all live in the Streamlit **"Cross-check with other models"** tab.

| Model | Question it answers | Tooling |
|---|---|---|
| **AlphaGenome** | does a *variant* change expression? | `co_scientist/variant_scorer.py`, score in Colab |
| **Boltz** | does a *molecule* bind the target? | `co_scientist/boltz_scorer.py`, plain API ($100 launch credits) |
| **Open Targets** | is the *gene* linked to the disease? | `co_scientist/target_scorer.py` — free GraphQL, no key |
| **DepMap** | is the *gene* a real dependency? | `co_scientist/target_scorer.py` — public CRISPRGeneEffect.csv |
| **AlphaMissense** | is a *coding variant* pathogenic? | `co_scientist/target_scorer.py` — free, via Ensembl VEP |

The gene judges (Open Targets / DepMap) share one gold set + merge step:

```bash
# Open Targets (free) + DepMap (needs data/depmap/CRISPRGeneEffect.csv; add
# Model.csv to restrict DepMap to multiple-myeloma cell lines):
python -m benchmark.build_target_scores       # writes opentargets_scores.json, depmap_scores.json
python -m benchmark.elo_vs_variant_score --scores benchmark/opentargets_scores.json
```

**AlphaMissense** scores *variants*, so it has its own real-data gold set —
pathogenic + benign missense variants pulled from ClinVar (coordinates read from
each record's canonical SPDI, never hand-typed):

```bash
python -m benchmark.fetch_clinvar             # real ClinVar missense variants -> clinvar_missense.json
python -m benchmark.build_missense_scores     # rank + score (Ensembl VEP) -> alphamissense_scores.json
python -m benchmark.elo_vs_variant_score --scores benchmark/alphamissense_scores.json
```

`build_missense_scores` also prints a calibration check (mean AlphaMissense for
pathogenic vs benign — they should separate), since each variant carries its
ClinVar answer.

**Boltz quick test** (binding):

```bash
# 1. sign up + redeem $100 credits (code BOLTZLAUNCH), make an API key:
#    https://api.boltz.bio/console/signup
export BOLTZ_API_KEY=...      # and ANTHROPIC_API_KEY for the Elo column
# 2. pull real UniProt sequences for the gold-set targets
python -m benchmark.fetch_uniprot
# 3. score the binding gold set + merge with Elo -> boltz_scores.json
python -m benchmark.build_boltz_scores
# 4. see the correlation (or use the app's Cross-check tab)
python -m benchmark.elo_vs_variant_score --scores benchmark/boltz_scores.json
```

⚠️ `fetch_uniprot.py` pulls the canonical reviewed human sequences, but the
ligand SMILES in `gold_set_binding.py` are illustrative — verify them before
trusting a number. The Boltz API is new: if a call errors, verify the endpoints
in `boltz_scorer.py` against the console's API reference.

### What the panel found (ERAD / bortezomib-resistant myeloma gold set)

Three independent judges, each scoring a different axis, all **disagree** with the
LLM Elo ranking — which is the point of running a panel rather than one judge:

| Judge | Axis | Spearman(Elo, model) |
|---|---|---|
| **AlphaGenome** | regulatory effect | **−0.60** |
| **Boltz** | binding confidence | **+0.26** |
| **Open Targets** | disease association | **−0.06** |
| **DepMap** | gene dependency | **−0.10** |
| **AlphaMissense** | variant pathogenicity | undefined (calibration +0.70) |

Open Targets and DepMap show the same near-perfect inversion: the LLM judge tops
the *elaborate* ERAD genes (SYVN1/HRD1, SEL1L, EDEM1); both external models put the
*proven* drug target (PSMB5, near the bottom of the Elo list) on top — Open Targets
by disease evidence, DepMap because the proteasome is an essential dependency. The
negative control (OR2T1) sits last in Open Targets and isn't in DepMap's CRISPR
library at all (so DepMap is n=5). n is small throughout, so read these as flags to
investigate, not verdicts.

AlphaMissense passed its own calibration (mean pathogenicity 0.93 for ClinVar
pathogenic vs 0.23 for benign — a +0.70 separation) but its Elo correlation is
*undefined*: the LLM gave all five variant hypotheses the same Elo, so there's
nothing to correlate. Phrased as "this variant is damaging," they read identically
to the judge — no LLM signal at the single-variant grain, while AlphaMissense
separates them cleanly.

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

## Two newer experiments

**Ontology discrimination (`run_adversarial.py`).** A plain "does grounding raise
Spearman" test is too easy — and risks rewarding a judge that just filters for
consensus, penalising novel hypotheses. So `gold_set_adversarial.py` has three
kinds: *solid* (correct), *novel* (real but cutting-edge, may not resolve in the
ontology), and *trap* (fluent but contradicts a canonical fact). The win
condition is **trap_demotion > 0** (grounding sinks the false ones) **with
novelty_penalty ≈ 0** (it leaves the novel ones alone). Needs OntoMCP + a key.

**Elo vs. an independent predictor (`elo_vs_variant_score.py`).** For hypotheses
you can frame as a regulatory variant, score them with a sequence-to-function
model and correlate against the Elo ranking. Low correlation = Elo isn't enough
for candidate selection on its own.

Scoring backends live in `co_scientist/variant_scorer.py`:
- **AlphaGenome** (recommended) — free API, no GPU. `pip install alphagenome`,
  get a non-commercial key at <https://www.alphagenomedocs.com/>, then
  `export ALPHAGENOME_API_KEY=...`.
- **Enformer** — fully open, runs locally on a GPU (Colab), via `enformer-pytorch`.

Both score *sequence/variant* claims, not perturbation claims ("inhibit p97"
isn't a sequence change) — apply them to the subset that fits.

## The one change to try first

In `co_scientist/graph.py`:

```python
from benchmark.fair_judge import ranking_fair
g.add_node("ranking", ranking_fair)   # was: agents.ranking
```

…then raise the match budget (see the plan). Re-run `validate` before and after.
