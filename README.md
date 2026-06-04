# Benchmate

A small AI Co-Scientist for biomedical hypothesis generation. Seven LangGraph
agents (Generation, Reflection, Ranking, Proximity, Evolution, Meta-review,
Supervisor) talk to each other in a loop, propose hypotheses, run a pairwise
Elo tournament, and refine the winners. PubMed is wired in as a real tool.
**Geneformer in-silico perturbation results are wired in as cached evidence**
so the agents can reason about your own experimental data, not just the
literature.

Designed to be readable in one sitting (~1,200 lines across the package and UI)
and extended one piece at a time.

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
│   ├── llm.py              # Claude wrapper with structured output
│   ├── agents.py           # the 7 agents (Generation reads Geneformer cache,
│   │                       # Reflection fact-checks against it)
│   └── graph.py            # LangGraph wiring of the supervisor loop
├── notebooks/              # Geneformer perturbation notebooks (Colab)
│   ├── 01_geneformer_erad_perturbation.ipynb        # first try (hepatocytes)
│   └── 02_geneformer_ciliated_cells.ipynb           # ciliated cells run
├── data/geneformer/        # cached perturbation results (CSV, gitignored)
│   └── README.md           # how to populate
└── ui/                     # Streamlit-based "user-friendly Benchmate"
    ├── app.py              # 3-tab Streamlit app
    ├── notebook_gen.py     # parameterise notebook 02 with user's genes
    ├── colab_handoff.py    # push notebook to GitHub Gist → Colab URL
    └── watcher.py          # watch Drive sync folder, copy CSVs into cache
```

## Two ways to use it

### A. CLI (the core loop, no UI)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set ANTHROPIC_API_KEY

python run.py "Find novel drug repurposing candidates for acute myeloid leukemia"
```

Iterations print top hypotheses + Elo ratings as they go. State is checkpointed
to `state.json`; resume with `python run.py --resume`.

### B. UI (gene input → Colab → cache → Benchmate, all in one app)

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`. Three tabs:

1. **New perturbation.** Type gene symbols (e.g. `TXNDC15, SYVN1`). Resolves
   them to Ensembl IDs via mygene, generates a parameterised copy of
   `notebooks/02_geneformer_ciliated_cells.ipynb` with your genes pre-filled,
   pushes it to a GitHub Gist, and hands you a one-click "Open in Colab" link.
2. **Inspect cache.** Browse what's already in `data/geneformer/` — pick a
   gene, see the top-N affected genes, sortable.
3. **Run Benchmate.** Paste a research goal, choose how many iterations, hit
   Run. Streams logs in the page. Downloads `state.json` when finished.

Optional companion: `python -m ui.watcher` runs a watchdog that copies
Colab's perturbation CSVs from your synced Google Drive folder into
`data/geneformer/` automatically, so you never touch the files by hand.

## The Geneformer integration

When the research goal (or a hypothesis under review) mentions a gene symbol
that has a cached `{GENE}_stats.csv` in `data/geneformer/`, the **Generation**
agent injects that gene's top-10 affected genes into its prompt as evidence,
and the **Reflection** agent uses the same evidence to fact-check proposed
mechanisms. The lookup is a pandas read; latency is milliseconds.

To populate the cache:
- Run `notebooks/02_geneformer_ciliated_cells.ipynb` in Colab with your
  chosen `TARGETS`, or use the UI's tab 1 to generate the notebook for you.
- Drop the resulting `*_stats.csv` files into `data/geneformer/`
  (manually, or let `ui/watcher.py` do it).

See `data/geneformer/README.md` for the expected CSV schema.

## What's still stubbed

- **BioNeMo NIM calls** (`tools.py::call_bionemo_nim`) — point at your endpoint
  when you have one.
- **Web search** — wire Tavily or Perplexity if you want non-PubMed evidence.
- **Vector embeddings** — Proximity agent uses a hash-based proxy; swap in
  `voyage-3` or `text-embedding-3-large` for real clustering.
- **Async parallelism** — supervisor runs one agent at a time for clarity.

## How to extend

Productive sequence, roughly in order of value:

1. **Run it once end-to-end** — both the CLI and the UI — before changing
   anything. Get a feel for what the agents produce.
2. **Add more genes to the cache** — every new gene you perturb makes the
   Generation agent strictly more useful for goals that mention it.
3. **Swap in a real embedding model** in `tools.py::embed` so the Proximity
   agent's clustering reflects real semantic similarity.
4. **Wire BioNeMo NIMs** — ESM-2 for protein embeddings or MolMIM for
   molecule generation. Function signature is ready in `tools.py`.
5. **Add LangSmith / NAT tracing** — set `LANGSMITH_API_KEY` and LangGraph
   traces automatically.

## Important honesty

This mirrors the *architecture* of Google's Co-Scientist, not its quality.
Google ran their version for many hours per research goal with Gemini 2.5
Pro and many tools. Benchmate's value isn't in matching that — it's in
giving you a loop you can fully reason about, then extending it with the
specific evidence sources your specific research actually needs. The
Geneformer wiring is the first such extension; your own RNA-seq results,
PPI data, or DepMap viability scores would slot in the same way.
