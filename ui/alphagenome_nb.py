"""Generate an AlphaGenome scoring notebook for Colab.

Same idea as notebook_gen.py, different model. AlphaGenome needs Python 3.10+
and a free API key, so Colab is the path of least resistance — but the template
notebook shipped with placeholder coordinates baked into its VARIANTS cell
(`pos=81_000_000`), while the real GTEx eQTL positions live in
benchmark/real_variant_coords.json (`pos=81499973`).

Scoring a placeholder position isn't a small error. AlphaGenome will happily
return a number for an arbitrary locus, and that number will be ~0, which reads
as "this variant does nothing" rather than "you scored the wrong base". So this
generator refuses to emit a notebook full of made-up coordinates: it injects the
real ones, records where each came from, and raises if they aren't available.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "notebooks" / "03_alphagenome_variant_scoring.ipynb"
REAL_COORDS = REPO / "benchmark" / "real_variant_coords.json"
OUT_DIR = REPO / "notebooks" / "generated"


def _variants_for_genes(genes: list[str]) -> tuple[list[dict], str]:
    """Fetch a real eQTL for each of `genes`, live from GTEx.

    This is the path that makes the notebook follow the user's actual question
    rather than the built-in ERAD gold set. Nothing is invented: a gene with no
    significant eQTL is dropped and named in the provenance line, because
    AlphaGenome genuinely has nothing to score there.
    """
    from benchmark.fetch_eqtls import fetch_for_genes

    coords, problems = fetch_for_genes(genes)
    rows = [{"label": f"{sym}_eqtl", "gene": sym, "tier": "eQTL", **c}
            for sym, c in coords.items()]
    if not rows:
        raise RuntimeError(
            "GTEx returned no usable eQTL for any of these genes, so there's "
            "nothing AlphaGenome can score.\n\n" + "\n".join(problems)
            + "\n\nAlphaGenome scores regulatory variants. If your question is "
              "about knockdowns or drug effects rather than variants, DepMap "
              "and Open Targets are the right models — not this one.")
    prov = (f"{len(rows)} variants, each the most significant GTEx eQTL "
            f"(hg38) for a gene from your question")
    if problems:
        prov += f". No eQTL available for: {'; '.join(problems)}"
    return rows, prov


def _load_variants() -> tuple[list[dict], str]:
    """Real variants to score, plus a one-line provenance string.

    Raises if only placeholders exist — better to send the user to
    `fetch_eqtls` than to hand them a notebook that quietly scores fiction.
    """
    from benchmark.gold_set_variants import GOLD_VARIANTS

    real: dict = {}
    if REAL_COORDS.exists():
        try:
            real = json.loads(REAL_COORDS.read_text())
        except Exception:
            real = {}

    rows, missing = [], []
    for g in GOLD_VARIANTS:
        c = real.get(g["label"])
        if not c:
            missing.append(g["label"])
            continue
        rows.append({"label": g["label"], "gene": g["gene"],
                     "tier": g.get("tier", "?"), **c})

    if not rows:
        raise RuntimeError(
            "No real variant coordinates found. The gold set ships with "
            "placeholder positions, and scoring those produces meaningless "
            "near-zero effects. Run `python -m benchmark.fetch_eqtls` first — "
            "it pulls genuine GTEx eQTLs into benchmark/real_variant_coords.json.")

    prov = (f"{len(rows)} variants with real GTEx eQTL coordinates "
            f"(hg38, from benchmark/real_variant_coords.json)")
    if missing:
        prov += f". Skipped {len(missing)} with placeholders only: {', '.join(missing)}"
    return rows, prov


def _variants_cell(rows: list[dict]) -> list[str]:
    src = ["# Real GTEx eQTL coordinates (hg38) — pulled by "
           "benchmark/fetch_eqtls.py.\n",
           "# Do NOT hand-edit these to 'tidy' numbers: AlphaGenome scores the\n",
           "# exact base you give it, and a rounded position is a different locus.\n",
           "VARIANTS = [\n"]
    w = max(len(r["label"]) for r in rows) + 3      # quotes + comma
    for r in rows:
        lab = f"{r['label']!r},"
        chrom = f"{r['chrom']!r},"
        pos = f"{r['pos']},"
        # carry the provenance inline: which SNP, which tissue, what p-value.
        # Without it there's no way to tell a real eQTL from a typo later.
        note = f"{r['tier']} · {r['gene']}"
        if r.get("rsid"):
            note += f" · {r['rsid']}"
        if r.get("tissue"):
            note += f" · {r['tissue']}"
        if r.get("pvalue") is not None:
            note += f" · p={r['pvalue']}"
        src.append(f"    dict(label={lab:<{w}} chrom={chrom:<10} "
                   f"pos={pos:<12} ref={r['ref']!r}, alt={r['alt']!r}),"
                   f"  # {note}\n")
    src.append("]\n")
    src.append("print(f'{len(VARIANTS)} variants to score')\n")
    return src


def generate_alphagenome_notebook(genes: list[str] | None = None,
                                  out_dir: Path | None = None
                                  ) -> tuple[Path, int]:
    """Write a Colab-ready AlphaGenome notebook. Returns (path, n_variants).

    Pass `genes` (from the research question / Start here plan / the hypothesis
    being cross-checked) and the notebook scores real GTEx eQTLs for those
    genes. With no genes it falls back to the built-in ERAD gold set, which is
    a benchmark, not your question.
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template notebook missing: {TEMPLATE}")

    genes = [g for g in (genes or []) if str(g).strip()]
    rows, prov = _variants_for_genes(genes) if genes else _load_variants()
    nb = json.loads(TEMPLATE.read_text())

    replaced = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if cell.get("cell_type") == "code" and "VARIANTS = [" in src:
            cell["source"] = _variants_cell(rows)
            replaced = True

        # the template warns that coordinates are placeholders — no longer true,
        # and leaving the warning in would train the reader to distrust real data
        elif cell.get("cell_type") == "markdown" and "placeholder" in src.lower():
            cell["source"] = [
                "### 3. The variants to score\n\n",
                f"{prov}.\n\n",
                "Each is a variant empirically associated with its gene's "
                "expression, so AlphaGenome is being asked a question it can "
                "actually answer.\n",
            ]

        # the export cell: make sure it says where the file goes next
        elif cell.get("cell_type") == "code" and "files.download" in src:
            if "Benchmate" not in src:
                cell["source"] = list(cell.get("source", [])) + [
                    "\nprint('Next: Benchmate -> Cross-check -> Calibration -> "
                    "AlphaGenome panel -> upload this file.')\n"]

    if not replaced:
        raise RuntimeError("Template has no `VARIANTS = [` cell to replace — "
                           "the notebook layout changed.")

    # generated notebooks ship clean: no stale outputs from someone else's run
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # and they parse, or we don't write them
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        clean = "\n".join("" if ln.strip().startswith(("!", "%")) else ln
                          for ln in code.split("\n"))
        try:
            compile(clean, f"<cell {i}>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Generated cell {i} is not valid Python: "
                             f"{e.msg} (line {e.lineno})") from e

    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = ("_".join(r["gene"] for r in rows[:4]) if genes else "erad_goldset")
    slug = re.sub(r"[^A-Za-z0-9_]+", "", slug) or "variants"
    path = out_dir / f"alphagenome_{slug}_{uuid.uuid4().hex[:6]}.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    return path, len(rows)


if __name__ == "__main__":
    p, n = generate_alphagenome_notebook()
    print(f"wrote {p} ({n} variants)")
