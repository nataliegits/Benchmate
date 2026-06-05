# Deploying Benchmate publicly

Two deployment paths, ranked by ease.

## Streamlit Community Cloud (free, recommended)

Streamlit's own hosting service. Free tier is generous, integrates directly
with GitHub, deploys in under five minutes.

### One-time setup

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   the GitHub account that owns this repo (`nataliegits`).
2. Click **New app**.
3. Fill in:
   - **Repository:** `nataliegits/Benchmate`
   - **Branch:** `main`
   - **Main file path:** `ui/app.py`
   - **App URL:** pick a slug — e.g. `benchmate.streamlit.app`
4. Click **Advanced settings** and add a secret (highly recommended):
   ```
   GITHUB_TOKEN = "ghp_..."
   ```
   Without this, public visitors get a "Download notebook → upload to
   Colab manually" two-step instead of the one-click Open in Colab
   button. Create a fine-grained PAT at
   [github.com/settings/tokens?type=beta](https://github.com/settings/tokens?type=beta)
   with **only** the `Gist: Read and write` scope. Generated notebooks
   will be public gists under your account.
5. Click **Deploy**.

The first build takes ~3 minutes (installing the requirements). After that,
every git push to `main` redeploys automatically.

### What users see

- The app's URL is public; anyone can visit it.
- The sidebar asks each user for their own **Anthropic API key**. Keys live
  only in the visitor's browser session — they are never written to disk or
  sent to the maintainer.
- The cache (`data/geneformer/`) is shared across visitors. Anyone can upload
  CSVs; everyone sees them in the sidebar list.
- The Gist push step (one-click Colab) is gracefully replaced by a
  "Download notebook" button when `gh` isn't available, which is the case
  on Streamlit Cloud.

### What users do

1. Enter their Anthropic key in the sidebar.
2. Tab 1: type genes + pick cell context → download the generated `.ipynb`.
3. Open [colab.research.google.com](https://colab.research.google.com) →
   File → Upload notebook → drop the file in.
4. In Colab: Runtime → Change runtime type → T4 GPU → Runtime → Run all.
   The final cell downloads each `*_stats.csv` to their browser.
5. Back in Benchmate sidebar: drag the downloaded CSVs into the **Upload
   CSVs** widget.
6. Tab 3: paste a research goal → Run Benchmate.

No Google Drive, no third-party hosting beyond Streamlit + Colab + Anthropic
(all of which the user owns separately).

### Limits to know

- Streamlit Community Cloud sleeps idle apps after a few hours of no traffic;
  first visitor after sleep waits ~30s for cold start.
- Free tier RAM: 1 GiB. Benchmate's loop fits easily; perturbation results
  (the cached CSVs) are ~1 MB each.
- Disk is ephemeral: uploaded CSVs persist for the lifetime of the running
  container but vanish on redeploy. For permanent shared state, either
  commit CSVs to the repo or back the cache with S3/GCS.

## Self-hosted (any VPS or container service)

Anything that can run a Python web app works. Examples:

- **Render** — free tier, autoscaling, Docker-friendly. Use
  `streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0`
  as the start command.
- **Railway** — pay-as-you-go, simple GitHub deploys. Similar start command.
- **Fly.io** — global edge deploys, Docker-based.
- **A VPS (Hetzner, DigitalOcean) under tmux** — `tmux new -s benchmate`,
  `streamlit run ui/app.py`, detach. Cheapest at scale; most manual.

For any of these you'll want to:

1. Set `ANTHROPIC_API_KEY` as a deployment secret (so the maintainer's key
   is the default fallback). Per-user keys still override via the sidebar.
2. Mount a persistent volume at `/app/data/geneformer/` if you want the
   shared cache to survive deploys.
3. Optionally put it behind Cloudflare Access or basic auth if you want
   to limit who can use your Anthropic credits.

## True one-click (the bigger lift)

If you want to eliminate the Colab step entirely — user clicks one button,
waits 30 min, gets hypotheses — the architecture is:

1. Host Geneformer on **Modal Labs** or **RunPod Serverless**. A `@stub.function`
   decorator wraps the perturbation in an HTTP endpoint.
2. Replace the "generate notebook" flow with a "run perturbation" button that
   POSTs to the Modal endpoint and polls for completion.
3. Results arrive as JSON, get parsed into the existing CSV schema, and feed
   straight into the agent loop.

Cost: ~$1–3 per perturbation in Modal credits (A10G or A100, depending on
gene count). Skip until there are real users willing to pay or your own
budget allows.
