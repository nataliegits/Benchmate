"""Thin wrapper around Claude with JSON-structured output.

This is the only file that talks to the Anthropic SDK. If you want to swap to
Gemini or a NIM-hosted Nemotron, replace just this file.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from anthropic import Anthropic

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


# Use the Claude SDK's enum-friendly model string. Swap to whatever model you
# prefer — Sonnet is the sweet spot of speed and reasoning quality.
DEFAULT_MODEL = os.environ.get("CO_SCIENTIST_MODEL", "claude-sonnet-4-6")


def call(prompt: str, *, system: str = "", model: str | None = None,
         max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """One-shot text completion. Returns the model's text response."""
    msg = _get_client().messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


def call_json(prompt: str, *, system: str = "", model: str | None = None,
              max_tokens: int = 2048, temperature: float = 0.7) -> Any:
    """Call the model and parse JSON out of its response.

    We instruct the model to wrap JSON in ```json fences, then strip them.
    Trades a small amount of robustness for not needing tool_use plumbing.
    """
    fenced_prompt = (prompt
                     + "\n\nReply with a valid JSON object wrapped in "
                       "```json ... ``` fences. Nothing else.")
    raw = call(fenced_prompt, system=system, model=model,
               max_tokens=max_tokens, temperature=temperature)

    match = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    payload = match.group(1).strip() if match else raw.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e
