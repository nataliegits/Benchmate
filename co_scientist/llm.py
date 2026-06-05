"""LLM wrapper with per-role model routing.

Uses `litellm` as the multi-provider backend so the same code can call
Anthropic, OpenAI, Google, Mistral, Bedrock, or anything else litellm
supports (15+ providers). The model picked depends on which agent role
is calling — see `llm_config.py` for the role→model mapping.

This is the only file that talks to the LLM SDK. If you want to swap the
backend (e.g. to Pi's Python SDK or a NIM-hosted Nemotron), replace just
this file.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import litellm

from .llm_config import model_for


# Don't log every call — Benchmate's own output is the source of truth
litellm.suppress_debug_info = True


# Backwards-compat: callers that don't pass a role get this default
DEFAULT_MODEL = os.environ.get("CO_SCIENTIST_MODEL", "anthropic/claude-sonnet-4-6")


def _resolve_model(role: str | None, explicit: str | None) -> str:
    """Pick the model for this call: explicit override > role lookup > default."""
    if explicit:
        return explicit
    if role:
        return model_for(role)
    return DEFAULT_MODEL


def call(prompt: str, *, system: str = "", role: str | None = None,
         model: str | None = None, max_tokens: int = 2048,
         temperature: float = 0.7) -> str:
    """One-shot text completion. Returns the model's text response.

    `role` picks the model via llm_config (e.g. "ranking" → Haiku).
    `model` overrides the role lookup with an explicit model string.
    """
    use_model = _resolve_model(role, model)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = litellm.completion(
        model=use_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_json(prompt: str, *, system: str = "", role: str | None = None,
              model: str | None = None, max_tokens: int = 2048,
              temperature: float = 0.7) -> Any:
    """Call the model and parse JSON out of its response.

    Instructs the model to wrap JSON in ```json fences. Falls back to
    brace-matched JSON when output is truncated and the closing fence is
    missing — `max_tokens` is the usual culprit.
    """
    fenced_prompt = (prompt
                     + "\n\nReply with a valid JSON object wrapped in "
                       "```json ... ``` fences. Nothing else.")
    raw = call(fenced_prompt, system=system, role=role, model=model,
               max_tokens=max_tokens, temperature=temperature)

    match = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if match:
        payload = match.group(1).strip()
    else:
        first = next((i for i, c in enumerate(raw) if c in "{["), -1)
        last = max(raw.rfind("}"), raw.rfind("]"))
        payload = (raw[first:last + 1].strip()
                   if first >= 0 and last > first else raw.strip())
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e
