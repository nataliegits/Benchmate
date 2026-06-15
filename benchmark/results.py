"""Tiny on-disk store for benchmark results.

The live benchmarks cost API calls and (for the ontology arm) need a running
OntoMCP server — neither of which exists on a public/hosted deployment. So the
pattern is: run the benchmarks **locally**, which appends the numbers here;
commit the JSON; the hosted Streamlit app then *displays* the saved results
instead of re-running them. No API key or OntoMCP needed in the cloud, and no
public visitor can spend your money.

Stdlib only. One JSON file per benchmark, holding a list of runs (newest last).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
KEEP = 20  # cap stored runs per benchmark


def _path(name: str) -> Path:
    return RESULTS_DIR / f"{name}.json"


def load_runs(name: str) -> list[dict]:
    """All saved runs for a benchmark (oldest first); [] if none/unreadable."""
    p = _path(name)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def latest(name: str) -> dict | None:
    runs = load_runs(name)
    return runs[-1] if runs else None


def save_run(name: str, payload: dict) -> dict:
    """Append a run and persist it. Returns the stored payload (with timestamp).

    `payload` convention (all optional except metrics):
        label   : human title
        params  : {slider/knob: value}      shown as a caption
        metrics : {name: display_string}    shown as st.metric tiles
        note    : one-line takeaway
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamped = {**payload,
               "captured": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    runs = load_runs(name)
    runs.append(stamped)
    _path(name).write_text(json.dumps(runs[-KEEP:], indent=2))
    return stamped
