# Benchmate

A runnable LangGraph skeleton of the Google Co-Scientist architecture, scoped for
biomedical / drug-discovery research. Seven agents, Elo tournament, PubMed tool.

This is intentionally small (~600 lines) so you can read it in one sitting and then
extend it. It's *not* production-ready — it's the day-one foundation the Co-Scientist
paper would have you build, minus the polish.

## What's in here

```
co_scientist_starter/
├── README.md
├── requirements.txt
├── .env.example
├── run.py                  # entry point: python run.py "your research goal"
└── co_scientist/
    ├── state.py            # shared state: hypotheses, Elo, memory
    ├── elo.py              # Elo math + tournament scheduler
    ├── tools.py            # PubMed E-utilities, web search stub, BioNeMo stub
    ├── llm.py              # Claude wrapper with structured output
    ├── agents.py           # the 7 agents (Generation, Reflection, ...)
    └── graph.py            # LangGraph wiring of the supervisor loop
```

## Quickstart

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. add API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

# 3. run
python run.py "Find novel drug repurposing candidates for acute myeloid leukemia"
```

Every iteration prints the current top hypotheses + their Elo ratings.
Ctrl-C to stop. State is checkpointed to `state.json` so you can resume.

## What it does (and what it doesn't, yet)

Implemented end-to-end:
- Supervisor agent that allocates work
- Generation, Reflection, Ranking (Elo), Proximity (embedding clusters),
  Evolution, Meta-review agents
- PubMed literature search as a real tool (no key required)
- Persistent JSON state, resumable

Stubbed / TODO for you:
- BioNeMo NIM calls (`tools.py::call_bionemo_nim`) — point at your endpoint
- Web search (currently disabled; plug in Tavily or Perplexity)
- Vector embeddings (currently uses a cheap hash-based proxy; swap in
  `voyage-3` or `text-embedding-3-large` for real similarity)
- The chat UI (this is CLI-only; wire NeMo Agent Toolkit or Streamlit on top)
- True async parallelism (the supervisor runs one agent at a time for clarity)

## How to extend

The most productive sequence:

1. **Run it as-is**, watch one full loop, read the printed hypotheses critically.
   Get a feel for what "good" output looks like before touching anything.
2. **Add a real embedding model** in `tools.py::embed`. Until you do, the
   Proximity agent's clustering is approximate.
3. **Wire BioNeMo NIMs.** Start with `meta/esm2nv` (cheap, useful for protein
   embeddings) or `nvidia/molmim-generate` (molecule generation). The function
   signature is already there in `tools.py`.
4. **Swap the LLM** if you want — `llm.py` is the only file that touches the
   Anthropic SDK. To use Gemini or a NIM-hosted Nemotron, change one file.
5. **Add LangSmith or NAT tracing.** Set `LANGSMITH_API_KEY` and LangGraph
   will trace automatically.

## Important honesty

This skeleton mirrors the *architecture* of Co-Scientist, not its quality.
Google ran their version for many hours per research goal with Gemini 2.0;
this starter is built to give you the loop running in 10 minutes so you can
iterate on the prompts, tools, and ranking logic — which is where most of
the quality actually comes from.
