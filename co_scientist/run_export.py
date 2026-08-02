"""Package a run as a zip: every file it touched, plus a readable PDF.

The point of the trace is that a run leaves something behind. A folder of JSONL
is fine for the app and useless for a colleague, so this bundles the whole run
into one file you can email:

    benchmate_<run_id>.zip
      report.pdf            the run as a document: question, evidence,
                            hypotheses and their reasoning, cross-checks,
                            design, protocol, bench result
      summary.txt           the same thing as plain text
      trace.jsonl           the raw event log, for diffing two runs
      meta.json
      files/                Geneformer CSVs, generated notebooks, state.json,
                            anything else the run referenced

PDF generation degrades rather than fails: without reportlab you still get the
zip, just with summary.txt instead of report.pdf. An export that refuses to run
because a formatting library is missing would be worse than a plain one.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from . import trace as _trace


def _styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm

    ss = getSampleStyleSheet()
    # Monochrome editorial, to match the app: Helvetica, generous leading,
    # no colour except the hairline rules.
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                                fontSize=20, leading=24, spaceAfter=2 * mm),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica",
                              fontSize=9.5, leading=13, textColor="#555555",
                              spaceAfter=6 * mm),
        "h": ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold",
                            fontSize=12, leading=15, spaceBefore=5 * mm,
                            spaceAfter=1.5 * mm),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=10, leading=14, spaceAfter=1.5 * mm),
        "kv": ParagraphStyle("kv", parent=ss["Normal"], fontName="Helvetica",
                             fontSize=9.5, leading=13, leftIndent=8,
                             textColor="#333333", spaceAfter=1),
        "mono": ParagraphStyle("m", parent=ss["Normal"], fontName="Courier",
                               fontSize=8.5, leading=11, textColor="#444444"),
    }


def _esc(s) -> str:
    """Escape for reportlab's mini-markup, which treats < and & as markup."""
    return (str(s or "").replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def build_pdf(run_id: str) -> bytes | None:
    """The run as a PDF. Returns None if reportlab isn't available."""
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.units import mm
        from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                        Spacer)
    except Exception:
        return None

    evs = _trace.read(run_id)
    if not evs:
        return None
    meta = {}
    try:
        meta = json.loads((_trace.folder(run_id) / "meta.json").read_text())
    except Exception:
        pass

    S = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER, title=f"Benchmate run {run_id}",
        author="Benchmate", leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm)

    story = [Paragraph("Benchmate run", S["title"])]
    story.append(Paragraph(
        f"{run_id} &nbsp;&middot;&nbsp; started {_esc(meta.get('started', '?'))}",
        S["sub"]))

    q = meta.get("question") or next(
        (e["detail"] for e in evs if e["step"] == "question"), "")
    if q:
        story.append(Paragraph("The question", S["h"]))
        story.append(HRFlowable(width="100%", thickness=0.6, color="#111111",
                                spaceAfter=3))
        story.append(Paragraph(_esc(q), S["body"]))

    # Group events by step, in loop order, so the document reads as a
    # narrative rather than a timestamped log.
    order = {s: i for i, s in enumerate(_trace.STEP_ORDER)}
    by_step: dict[str, list] = {}
    for e in evs:
        if e["step"] == "question":
            continue
        by_step.setdefault(e["step"], []).append(e)

    for step in sorted(by_step, key=lambda s: order.get(s, 99)):
        story.append(Paragraph(_trace.STEP_LABEL.get(step, step), S["h"]))
        story.append(HRFlowable(width="100%", thickness=0.6, color="#111111",
                                spaceAfter=3))
        for e in by_step[step]:
            story.append(Paragraph(f"<b>{_esc(e['headline'])}</b>", S["body"]))
            if e.get("detail"):
                story.append(Paragraph(_esc(e["detail"]), S["kv"]))
            for k, v in (e.get("outputs") or {}).items():
                story.append(Paragraph(
                    f"<b>{_esc(k)}:</b> {_esc(v)}", S["kv"]))
            if e.get("files"):
                story.append(Paragraph(
                    "files: " + _esc(", ".join(e["files"])), S["mono"]))
            story.append(Spacer(1, 3))

    files = sorted(p.name for p in (_trace.folder(run_id) / "files").glob("*")) \
        if (_trace.folder(run_id) / "files").exists() else []
    if files:
        story.append(Paragraph("Files in this run", S["h"]))
        story.append(HRFlowable(width="100%", thickness=0.6, color="#111111",
                                spaceAfter=3))
        for f in files:
            story.append(Paragraph(_esc(f), S["mono"]))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "Generated by Benchmate. Hypotheses are ranked by a language model and "
        "checked against independent models; neither is a substitute for the "
        "bench result.", S["sub"]))

    doc.build(story)
    return buf.getvalue()


def build_zip(run_id: str, extra_dirs: dict[str, Path] | None = None) -> bytes:
    """The whole run as a zip. `extra_dirs` maps a folder name in the archive
    to a directory to include (used to sweep in the Geneformer cache and any
    generated notebooks that weren't explicitly attached to an event)."""
    folder = _trace.folder(run_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        pdf = build_pdf(run_id)
        if pdf:
            z.writestr("report.pdf", pdf)
        z.writestr("summary.txt", _trace.summarise(run_id))
        for name in ("trace.jsonl", "meta.json"):
            p = folder / name
            if p.exists():
                z.write(p, name)
        fdir = folder / "files"
        if fdir.exists():
            for p in sorted(fdir.glob("*")):
                if p.is_file():
                    z.write(p, f"files/{p.name}")
        seen = {p.name for p in fdir.glob("*")} if fdir.exists() else set()
        for arc, d in (extra_dirs or {}).items():
            if not d or not Path(d).exists():
                continue
            for p in sorted(Path(d).rglob("*")):
                if p.is_file() and p.name not in seen:
                    z.write(p, f"{arc}/{p.name}")
    return buf.getvalue()
