"""Freezer bridge — turn a CryoVision box scan into "where is reagent X?"

CryoVision (separate repo) reads a photo of a 10x10 cryobox and writes a map of
every slot: `position,row,column,label` (CSV) or `{position: label}` (JSON).
Benchmate consumes that map so the co-scientist can locate a reagent it wants to
test — the "find" edge of the loop.

We read CryoVision's output file directly rather than importing its OpenCV /
RF-DETR / vision stack, so Benchmate stays light. To produce a fresh map, run
CryoVision on a photo:  `python cryovision.py --image box.jpg --output box.csv`.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

FREEZER_DIR = Path(__file__).resolve().parent.parent / "data" / "freezer"
DEFAULT_BOX = FREEZER_DIR / "demo_drug_box.csv"


def load_box(path: str | Path) -> list[dict]:
    """Load a CryoVision box map (CSV or JSON) into [{position,row,column,label}]."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        out = []
        for pos, lab in data.items():
            m = re.match(r"([A-Za-z]+)(\d+)", pos)
            out.append({"position": pos,
                        "row": m.group(1) if m else "",
                        "column": m.group(2) if m else "",
                        "label": (lab or "")})
        return out
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"position": r.get("position", ""),
                         "row": r.get("row", ""),
                         "column": r.get("column", ""),
                         "label": (r.get("label") or "").strip()})
    return rows


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", s.lower())


def locate(reagent: str, box: list[dict]) -> list[dict]:
    """Slots whose label matches `reagent`, best first. Exact > substring > token.
    Each hit is the cell dict plus a `score` (3/2/1). Empty list if not found."""
    q = _norm(reagent)
    toks = [_norm(t) for t in reagent.split() if len(t) >= 3]
    hits = []
    for cell in box:
        lab = cell.get("label", "")
        if not lab:
            continue
        n = _norm(lab)
        if q and q == n:
            s = 3
        elif q and (q in n or n in q):
            s = 2
        elif toks and any(t in n for t in toks):
            s = 1
        else:
            s = 0
        if s:
            hits.append((s, cell))
    hits.sort(key=lambda x: -x[0])
    return [dict(c, score=s) for s, c in hits]


def load_inventory(path: str | Path) -> list[dict]:
    """Load an inventory from a CryoVision box map (CSV/JSON) OR a free-form
    reagent list (CSV/Excel). Flexible columns: a name column (label / reagent /
    name / item / compound / antibody) and an optional location column
    (position / location / slot / box / freezer / where). Returns the same
    [{position,row,column,label}] shape that locate() / reconcile() consume."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".json":
        return load_box(p)
    if suf in (".xlsx", ".xls"):
        import pandas as pd
        recs = pd.read_excel(p).fillna("").to_dict("records")
    else:
        with open(p, newline="") as f:
            recs = [{k: (v or "") for k, v in r.items()} for r in csv.DictReader(f)]
    if not recs:
        return []
    keys = {str(k).lower().strip(): k for k in recs[0].keys()}

    def pick(*names):
        return next((keys[n] for n in names if n in keys), None)

    name_col = pick("label", "reagent", "name", "item", "compound", "antibody")
    loc_col = pick("position", "location", "slot", "box", "freezer", "where")
    out = []
    for r in recs:
        label = str(r.get(name_col, "")).strip() if name_col else ""
        if not label:
            continue
        pos = str(r.get(loc_col, "")).strip() if loc_col else ""
        m = re.match(r"([A-Za-z]+)?(\d+)?", pos)
        out.append({"position": pos,
                    "row": (m.group(1) or "") if m else "",
                    "column": (m.group(2) or "") if m else "",
                    "label": label})
    return out


def reconcile(reagents_needed: list[str], box: list[dict]) -> list[dict]:
    """For each needed reagent, is it in the box and where? Returns
    [{reagent, found, position, label}] — the have/need/where check for the
    'execute the experiment' step."""
    out = []
    for r in reagents_needed:
        hits = locate(r, box)
        if hits:
            out.append({"reagent": r, "found": True,
                        "position": hits[0]["position"], "label": hits[0]["label"]})
        else:
            out.append({"reagent": r, "found": False, "position": None, "label": None})
    return out


def available_boxes() -> list[str]:
    if not FREEZER_DIR.exists():
        return []
    return sorted(str(p) for p in FREEZER_DIR.glob("*.csv")) + \
        sorted(str(p) for p in FREEZER_DIR.glob("*.json"))
