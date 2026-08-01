"""Record what Benchmate did, step by step, into a run folder.

The complaint this answers: Benchmate makes a series of consequential choices —
which genes to perturb, which hypothesis won, which model disagreed, what the
bench said — and then shows you only the final state. You can't see how it got
there, you can't tell what changed when you swapped a gene, and afterwards
there's nothing to hand anybody.

So every step appends an event to `runs/<run_id>/trace.jsonl`, and every file it
touches gets copied into that same folder. One run, one directory, in order.

Design choices worth knowing:

* Append-only JSONL, not a database. A crashed run still leaves a readable
  trace, and you can tail it while a run is going.
* Events carry `inputs` and `outputs` separately, so a diff between two runs
  shows what actually changed rather than just what happened.
* Recording never raises. A trace is observability; if it breaks it must not
  take the science down with it.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

# The steps of the loop, in the order they belong on screen. Anything not
# listed still records; it just sorts last.
STEP_ORDER = [
    "question", "plan", "evidence", "notebook", "cache",
    "generate", "rank", "crosscheck", "design", "reagents",
    "bench", "feedback", "export",
]

STEP_LABEL = {
    "question": "Research question",
    "plan": "Plan",
    "evidence": "Evidence",
    "notebook": "Perturbation notebook",
    "cache": "Perturbation data",
    "generate": "Hypotheses generated",
    "rank": "Tournament",
    "crosscheck": "Cross-check",
    "design": "Experiment design",
    "reagents": "Reagents",
    "bench": "Bench result",
    "feedback": "Hypothesis updated",
    "export": "Export",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run(question: str = "") -> str:
    """Start a run and return its id. Ids sort chronologically by name."""
    rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    try:
        d = RUNS / rid
        (d / "files").mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(
            {"run_id": rid, "started": _now(), "question": question}, indent=2))
        if question:
            record(rid, "question", "You asked", detail=question)
    except Exception:
        pass
    return rid


def _path(run_id: str) -> Path:
    return RUNS / run_id / "trace.jsonl"


def record(run_id: str, step: str, headline: str, *, detail: str = "",
           inputs: dict | None = None, outputs: dict | None = None,
           files: list[str] | None = None) -> None:
    """Append one event.

    `headline` is the one line a human reads. `detail` is the sentence under it.
    `inputs`/`outputs` are for diffing runs. `files` are copied into the run
    folder so the record stays valid even if the original moves.
    """
    if not run_id:
        return
    try:
        d = RUNS / run_id
        (d / "files").mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files or []:
            try:
                src = Path(f)
                if src.exists() and src.is_file():
                    dst = d / "files" / src.name
                    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(src, dst)
                    saved.append(src.name)
            except Exception:
                pass
        ev = {"t": _now(), "step": step, "headline": headline,
              "detail": detail, "inputs": inputs or {},
              "outputs": outputs or {}, "files": saved}
        with (d / "trace.jsonl").open("a") as fh:
            fh.write(json.dumps(ev) + "\n")
    except Exception:
        pass          # observability must never break the run


def read(run_id: str) -> list[dict]:
    """Every event for a run, in the order recorded."""
    p = _path(run_id)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def runs() -> list[dict]:
    """All runs, newest first, with enough detail for a picker."""
    if not RUNS.exists():
        return []
    out = []
    for d in sorted(RUNS.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta = {}
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception:
            meta = {"run_id": d.name}
        evs = read(d.name)
        meta["n_events"] = len(evs)
        meta["steps"] = sorted({e["step"] for e in evs},
                               key=lambda s: STEP_ORDER.index(s)
                               if s in STEP_ORDER else 99)
        meta.setdefault("question", "")
        out.append(meta)
    return out


def folder(run_id: str) -> Path:
    return RUNS / run_id


def latest_run_id() -> str | None:
    r = runs()
    return r[0]["run_id"] if r else None


def summarise(run_id: str) -> str:
    """The run as plain text: what was asked, what happened, what came out.

    Deliberately readable on its own, so it can be pasted into an email or a
    lab notebook without Benchmate in front of you.
    """
    evs = read(run_id)
    if not evs:
        return "No trace recorded for this run."
    meta = {}
    try:
        meta = json.loads((RUNS / run_id / "meta.json").read_text())
    except Exception:
        pass
    lines = [f"Benchmate run {run_id}",
             f"Started {meta.get('started', '?')}", ""]
    if meta.get("question"):
        lines += [f"Question: {meta['question']}", ""]
    for e in evs:
        lines.append(f"[{e['t'][11:19]}] {STEP_LABEL.get(e['step'], e['step'])}: "
                     f"{e['headline']}")
        if e.get("detail"):
            lines.append(f"    {e['detail']}")
        for k, v in (e.get("outputs") or {}).items():
            lines.append(f"    {k}: {v}")
        if e.get("files"):
            lines.append(f"    files: {', '.join(e['files'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    rid = new_run("What ERAD-pathway genes drive bortezomib resistance in myeloma?")
    record(rid, "plan", "Picked 4 genes to perturb",
           detail="From the question, in plasma cells.",
           outputs={"genes": "SELENOS, DERL1, VCP, SYVN1"})
    record(rid, "generate", "12 hypotheses proposed",
           outputs={"top": "Elevated SEL1L sustains ERAD throughput"})
    print(summarise(rid))
    print("\nfolder:", folder(rid))
