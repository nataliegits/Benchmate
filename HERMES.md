# Running Benchmate behind Hermes

[Hermes](https://hermes-agent.nousresearch.com/) is an autonomous agent that
runs unattended on a server, accepts messages from Telegram/Discord/Slack/Email/CLI,
remembers what it learns, and can schedule automations. Wrapping Benchmate behind
Hermes lets you trigger research runs from chat, or schedule recurring runs that
quietly accumulate hypotheses while you sleep.

## What you get

Three usage patterns once Hermes is wired in:

1. **Chat-driven runs.** On Slack: *"hey hermes, run benchmate on FOXP3 in
   plasma cells, prioritise CAR-T relevance."* Hermes parses, calls
   `run_benchmate(...)`, posts the top hypothesis back to the channel.
2. **Scheduled runs.** *"every sunday at 10am, run benchmate on the three
   ERAD genes I haven't perturbed yet."* Hermes's natural-language cron
   does the scheduling; the runner does the work.
3. **Conversational cache.** *"hermes, what new hits has benchmate produced
   for E3 ligases this month?"* Hermes's memory queries the saved
   `state.json` history.

## What's in this repo

- `hermes/benchmate_runner.py` — JSON-in / JSON-out wrappers around the
  three operations Hermes needs: `list_cache`, `add_perturbation`,
  `run_benchmate`. Also runnable as `python -m hermes.benchmate_runner ...`
  for the chat-shell integration path.
- `hermes/__init__.py` — package marker.

## Deployment in three steps

### Step 1 — Get a server

Cheapest reasonable option is a Hetzner CX22 (€4/mo, 4 GB RAM, Frankfurt) or
a DigitalOcean Basic Droplet ($6/mo). Anything with Ubuntu 22.04+ and SSH access
works. You'll want at least 4 GB RAM because Hermes plus Benchmate's runtime
plus litellm's dependencies add up.

```bash
# from your Mac
ssh root@your.server.ip
```

### Step 2 — Install Hermes and Benchmate side-by-side

```bash
# Hermes (their canonical install)
curl -sSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes setup    # interactive — pick a chat platform, paste API keys

# Benchmate
git clone https://github.com/nataliegits/Benchmate ~/Benchmate
cd ~/Benchmate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Smoke-test the runner directly
python -m hermes.benchmate_runner list-cache
# expected: {"cached_genes": []}  (empty until you upload CSVs)
```

### Step 3 — Register Benchmate as a Hermes skill

Hermes auto-generates skills from natural-language examples. You teach it
once and it remembers:

> "hey hermes, you can run benchmate by calling
> `cd /root/Benchmate && source .venv/bin/activate && python -m hermes.benchmate_runner run "$GOAL" --max-iterations 8`.
> when someone asks you to generate hypotheses about a gene, use this."

Hermes will generate a skill manifest and persist it. From that point onward,
any chat message that asks for hypotheses triggers a Benchmate run.

For more control, write the skill yourself — Hermes uses standard YAML-style
skill manifests. The minimal one:

```yaml
name: benchmate_run
description: Generate testable biomedical hypotheses for a research goal
trigger: |
  Match messages that ask to generate hypotheses, propose experiments, or
  run benchmate. Extract the research goal as `goal`.
command: |
  cd /root/Benchmate
  source .venv/bin/activate
  python -m hermes.benchmate_runner run "{{ goal }}" --max-iterations 8
output: |
  Parse the JSON. Post the top hypothesis's statement and experiment to
  the channel, with the Elo score as a quality indicator.
```

Save as `~/.hermes/skills/benchmate_run.yaml` and reload Hermes (`hermes
reload`).

## How the runner is wired

`hermes/benchmate_runner.py` exposes three Python functions:

```python
list_cache()
    → {"cached_genes": ["TXNDC15", "SYVN1", "MARCHF6"]}

gene_neighbors("TXNDC15", top_n=10)
    → {"gene_symbol": "TXNDC15",
       "affected_genes": [{"symbol": "TOMM7", "cosine_shift": 0.021, ...}, ...]}

add_perturbation(["FOXP3"], cell_context="Plasma cells (extreme ER stress)")
    → {"notebook_path": "notebooks/generated/...ipynb",
       "resolved": {"FOXP3": "ENSG00000049768"},
       "next_step": "Open in Colab, Run All, drop CSV in data/geneformer/"}

run_benchmate("Generate hypotheses for FOXP3 in plasma cells", max_iterations=8)
    → {"iterations_run": 8,
       "top_hypotheses": [{"elo": 1287.4, "statement": "...", "experiment": "..."},
                          ...up to 5...]}
```

Each returns plain JSON-serializable dicts, so Hermes can hand them straight
into its chat formatter.

## Cost considerations

Hermes itself is free (MIT open source). The costs:

- **VPS:** $5–6/mo
- **Anthropic API:** the same $1–3 per Benchmate run as before. Set
  `ANTHROPIC_API_KEY` (and any other provider keys you've configured via
  litellm role routing) in `~/.hermes/.env`.

If you don't want to fund every chat-triggered run, add a `--dry-run` flag
in your skill manifest that returns the cache hit + a confirmation prompt
before actually invoking the model.

## When this is worth it

You should set up Hermes when:

- You're running Benchmate on >5 different research goals per week
- You want hypothesis generation to happen in the background while you do
  wet-lab work
- You want a chat interface (Slack / Discord) instead of a browser tab
- You have collaborators who should be able to trigger runs without learning
  Streamlit

You can skip Hermes when:

- You're still iterating on the prompts (the Streamlit UI gives faster
  feedback)
- You only run Benchmate occasionally
- You don't have a server you're comfortable maintaining

## The bigger picture: Pi + Hermes together

[Pi](https://pi.dev) handles multi-model routing inside Benchmate (cheaper
models for high-throughput agent roles — see `co_scientist/llm_config.py`).
Hermes handles the orchestration around Benchmate (chat triggers, schedules,
persistent state). They sit at different layers; using both gives you a
research workflow that's both cheap-per-run and trivial-to-trigger.
