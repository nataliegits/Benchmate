"""Accumulate cross-check scores from your own leaderboard.

The calibration panels correlate Elo against a model score across a *gold set*:
a fixed list of hypotheses with known-ish answers, scored once by a benchmark
script. That answers "is the judge trustworthy in general?"

It does not touch the hypotheses you actually cross-check in the app, which is
confusing: you score SEL1L live, open Calibration, and see SYVN1 and PSMB5 from
a benchmark run weeks ago. Same models, different data, no overlap.

So this records every live score alongside the Elo of the hypothesis it came
from, building a calibration set out of your own run. Once three or more
hypotheses have been scored by the same model, the correlation becomes a real
statement about *your* ranking rather than a generic benchmark.

Files are `benchmark/live_{model}_scores.json`, in the same
`[{label, elo, score}]` shape the existing panels already read.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DIR = Path(__file__).resolve().parent
MIN_POINTS = 3          # below this a correlation is noise


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model.lower())


def path_for(model: str) -> Path:
    return DIR / f"live_{_slug(model)}_scores.json"


def load(model: str) -> list[dict]:
    p = path_for(model)
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text())
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def record(model: str, label: str, elo: float | None, score: float | None,
           target: str = "") -> None:
    """Add or update one (hypothesis, score) point for `model`.

    Keyed on the hypothesis label, so re-running a model on the same hypothesis
    corrects the old point instead of double-counting it. Rows without an Elo
    are skipped: a score with no ranking to compare against contributes nothing
    to a correlation.
    """
    if elo is None or score is None or not label:
        return
    rows = [r for r in load(model) if r.get("label") != label]
    rows.append({"label": label, "elo": float(elo), "score": float(score),
                 "target": target})
    try:
        path_for(model).write_text(json.dumps(rows, indent=2))
    except Exception:
        pass          # never let bookkeeping break a scoring run


def elo_for_statement(statement: str, state_file: Path | None = None
                      ) -> float | None:
    """Elo of the hypothesis whose statement matches `statement`.

    Matches on a normalised prefix, because the text shown in a dropdown may be
    truncated relative to what's in state.json.
    """
    state_file = state_file or DIR.parent / "state.json"
    if not state_file.exists() or not statement:
        return None
    try:
        hyps = json.loads(state_file.read_text()).get("hypotheses", [])
    except Exception:
        return None

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "")).strip().lower()

    want = norm(statement)
    for h in hyps:
        got = norm(h.get("statement", ""))
        if got == want or got.startswith(want[:80]) or want.startswith(got[:80]):
            return float(h.get("elo", 0)) or None
    return None


def summary() -> list[dict]:
    """What's accumulated so far, for a status line in the UI."""
    out = []
    for model in ("Open Targets", "DepMap", "AlphaMissense", "AlphaGenome", "Boltz"):
        rows = load(model)
        if rows:
            out.append({"model": model, "n": len(rows),
                        "ready": len(rows) >= MIN_POINTS})
    return out
