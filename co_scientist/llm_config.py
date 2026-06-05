"""Per-role model routing for Benchmate.

Each agent in the loop has a different cost/quality tradeoff. The default
mapping below sends:

  - High-reasoning roles (generation, reflection, evolution) to Claude
    Sonnet — these are where the Geneformer evidence injection matters and
    where hypothesis quality lives.
  - Throughput roles (ranking, meta_review, supervisor) to Haiku 4.5 — fast,
    cheap, perfectly capable of pairwise-judgment work.

Override any role via env var: BENCHMATE_MODEL_<ROLE_UPPER>.
Override the default for unlisted roles via BENCHMATE_MODEL_DEFAULT.

Models use litellm's provider-prefixed naming, identical to Pi's:
  "anthropic/claude-sonnet-4-6"
  "anthropic/claude-haiku-4-5"
  "gemini/gemini-2.5-flash"
  "openai/gpt-5-mini"
  ... etc.
"""
from __future__ import annotations

import os


# Default role → model mapping. Tune these to your cost/quality preferences.
_DEFAULT_ROLE_MODELS: dict[str, str] = {
    "generation":  "anthropic/claude-sonnet-4-6",   # reads Geneformer evidence
    "reflection":  "anthropic/claude-sonnet-4-6",   # fact-checks against evidence
    "evolution":   "anthropic/claude-sonnet-4-6",   # writes long experiments
    "ranking":     "anthropic/claude-haiku-4-5",    # 30+ pairwise judgments/run
    "meta_review": "anthropic/claude-haiku-4-5",    # synthesizes patterns
    "supervisor":  "anthropic/claude-haiku-4-5",    # picks the next action
}

_FALLBACK_MODEL = "anthropic/claude-sonnet-4-6"


def model_for(role: str) -> str:
    """Return the model litellm should use for `role`.

    Resolution order:
      1. BENCHMATE_MODEL_<ROLE_UPPER> env var (per-role override)
      2. _DEFAULT_ROLE_MODELS[role]
      3. BENCHMATE_MODEL_DEFAULT env var (catch-all override)
      4. _FALLBACK_MODEL
    """
    env_specific = os.environ.get(f"BENCHMATE_MODEL_{role.upper()}")
    if env_specific:
        return env_specific
    if role in _DEFAULT_ROLE_MODELS:
        return _DEFAULT_ROLE_MODELS[role]
    return os.environ.get("BENCHMATE_MODEL_DEFAULT", _FALLBACK_MODEL)
