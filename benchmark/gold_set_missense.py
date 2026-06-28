"""Variant-level gold set for the AlphaMissense cross-check.

AlphaMissense scores how likely a *coding (missense) variant* is pathogenic — a
different axis from the gene-level judges. So this gold set is one hypothesis per
variant: "this missense change is damaging / function-altering." We rank these
with the LLM judge (Elo) and cross-check against AlphaMissense.

The variants are REAL ClinVar records fetched by `fetch_clinvar.py` (run that
first). Each carries its ClinVar classification, so we get a bonus: a known
answer. Tier reflects that classification:
  A  ClinVar pathogenic   (AlphaMissense should score high)
  C  ClinVar benign        (AlphaMissense should score low)

That makes this set both an Elo cross-check AND a calibration check on
AlphaMissense itself (does pathogenic separate from benign?).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_VARIANTS = HERE / "clinvar_missense.json"


def load_variants() -> list[dict]:
    """Real ClinVar missense variants from fetch_clinvar.py, with a label + tier."""
    if not _VARIANTS.exists():
        raise SystemExit("No clinvar_missense.json — run `python -m benchmark.fetch_clinvar` first.")
    rows = json.loads(_VARIANTS.read_text())
    out = []
    for r in rows:
        tier = "A" if r["clinsig"] == "pathogenic" else "C"
        label = f"{r['gene']}_{r['chrom']}-{r['pos']}{r['ref']}>{r['alt']}"
        out.append({**r, "tier": tier, "label": label})
    return out


def missense_hypotheses() -> list[dict]:
    """Frame each variant as a rankable hypothesis for the LLM judge."""
    hyps = []
    for v in load_variants():
        name = v.get("name") or f"{v['gene']} {v['chrom']}:{v['pos']}{v['ref']}>{v['alt']}"
        hyps.append(dict(
            label=v["label"], gene=v["gene"], clinsig=v["clinsig"],
            chrom=v["chrom"], pos=v["pos"], ref=v["ref"], alt=v["alt"],
            statement=(f"The {name} missense variant is a damaging, "
                       f"function-altering coding change in {v['gene']}."),
            rationale=("Coding missense variant; the claim is that the amino-acid "
                       "substitution disrupts protein function."),
        ))
    return hyps
