"""Fetch REAL missense variants (with GRCh38 coordinates) from ClinVar.

AlphaMissense scores a single coding variant's pathogenicity — so the AlphaMissense
cross-check needs real variants with real coordinates. We never hand-type those.
This pulls them from ClinVar via NCBI E-utilities (JSON), taking a few
*pathogenic* and a few *benign* missense SNVs for each gold-set gene, and reads
the genomic coordinates straight from each record's **canonical SPDI**
(e.g. "NC_000017.11:7676153:G:A"), which is unambiguous GRCh38.

Output -> benchmark/clinvar_missense.json
    [{gene, clinsig, chrom, pos, ref, alt, name}, ...]

clinsig is normalised to "pathogenic" or "benign" (the known answer), so the
AlphaMissense scores can be checked against ground truth as well as against Elo.

    python -m benchmark.fetch_clinvar
    python -m benchmark.fetch_clinvar --genes TP53,VCP --per-class 3

NCBI etiquette: set NCBI_EMAIL (and optionally NCBI_API_KEY) to raise rate limits.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HERE = Path(__file__).resolve().parent
OUT = HERE / "clinvar_missense.json"

# Default to the gold-set genes so the demo stays in-domain. Genes with no
# qualifying ClinVar variant are simply skipped (printed), so it's safe to widen.
DEFAULT_GENES = ["VCP", "PSMB5", "SYVN1", "SEL1L", "EDEM1", "XBP1"]

# NCBI RefSeq chromosome accessions -> chromosome name
_NC = {f"NC_0000{n:02d}": str(n) for n in range(1, 23)}
_NC["NC_000023"] = "X"
_NC["NC_000024"] = "Y"


def _params(extra: dict) -> dict:
    p = {"tool": "benchmate", "email": os.environ.get("NCBI_EMAIL", "benchmate@example.com")}
    if os.environ.get("NCBI_API_KEY"):
        p["api_key"] = os.environ["NCBI_API_KEY"]
    p.update(extra)
    return p


def _esearch(client, term: str, retmax: int) -> list[str]:
    r = client.get(f"{EUTILS}/esearch.fcgi",
                   params=_params({"db": "clinvar", "term": term,
                                   "retmode": "json", "retmax": retmax}))
    r.raise_for_status()
    return (((r.json() or {}).get("esearchresult") or {}).get("idlist")) or []


def _esummary(client, uids: list[str]) -> dict:
    if not uids:
        return {}
    r = client.get(f"{EUTILS}/esummary.fcgi",
                   params=_params({"db": "clinvar", "id": ",".join(uids),
                                   "retmode": "json"}))
    r.raise_for_status()
    return (r.json() or {}).get("result") or {}


def _spdi_to_coords(spdi: str):
    """'NC_000017.11:7676153:G:A' -> ('17', 7676154, 'G', 'A').  SPDI position is
    0-based; VEP/region wants 1-based, so +1. Only clean SNVs are returned."""
    try:
        acc_v, pos0, ref, alt = spdi.split(":")
        acc = acc_v.split(".")[0]
        chrom = _NC.get(acc)
        if not chrom or len(ref) != 1 or len(alt) != 1:
            return None
        return chrom, int(pos0) + 1, ref, alt
    except Exception:
        return None


def _clinsig_of(rec: dict) -> str:
    for key in ("germline_classification", "clinical_significance", "clinical_impact_classification"):
        desc = ((rec.get(key) or {}).get("description") or "").lower()
        if desc:
            return desc
    return ""


def _variants_for(client, gene: str, clinsig: str, per_class: int) -> list[dict]:
    """clinsig is 'pathogenic' or 'benign' — used both in the query and as the label."""
    term = (f'{gene}[gene] AND "missense variant"[molecular consequence] '
            f'AND "single nucleotide variant"[Type of variation] '
            f'AND "clinsig {clinsig}"[Properties] AND "GRCh38"[Assembly]')
    uids = _esearch(client, term, retmax=per_class * 4)   # over-fetch; many lack clean SPDI
    time.sleep(0.34)
    result = _esummary(client, uids)
    time.sleep(0.34)
    out = []
    for uid in uids:
        rec = result.get(uid) or {}
        vset = (rec.get("variation_set") or [{}])[0]
        coords = _spdi_to_coords(vset.get("canonical_spdi") or "")
        sig = _clinsig_of(rec)
        # keep only records whose own classification agrees (avoid conflicting/VUS)
        if not coords or clinsig not in sig or "conflicting" in sig:
            continue
        chrom, pos, ref, alt = coords
        out.append(dict(gene=gene, clinsig=clinsig, chrom=chrom, pos=pos,
                        ref=ref, alt=alt, name=vset.get("variation_name", "")))
        if len(out) >= per_class:
            break
    return out


def fetch(genes: list[str], per_class: int) -> list[dict]:
    import httpx
    rows: list[dict] = []
    with httpx.Client(timeout=30.0, headers={"User-Agent": "benchmate/1.0"}) as client:
        for gene in genes:
            got = []
            for clinsig in ("pathogenic", "benign"):
                try:
                    v = _variants_for(client, gene, clinsig, per_class)
                except Exception as e:
                    print(f"  {gene} ({clinsig}): error {e}")
                    v = []
                got.extend(v)
            print(f"  {gene:7}: {len(got)} variant(s) "
                  f"[{sum(r['clinsig']=='pathogenic' for r in got)} path / "
                  f"{sum(r['clinsig']=='benign' for r in got)} benign]")
            rows.extend(got)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", default=",".join(DEFAULT_GENES),
                    help="comma-separated gene symbols")
    ap.add_argument("--per-class", type=int, default=2,
                    help="max pathogenic and max benign per gene")
    args = ap.parse_args()
    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    print(f"Fetching ClinVar missense variants for: {', '.join(genes)}")
    rows = fetch(genes, args.per_class)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\n-> wrote {OUT.name} ({len(rows)} variants: "
          f"{sum(r['clinsig']=='pathogenic' for r in rows)} pathogenic, "
          f"{sum(r['clinsig']=='benign' for r in rows)} benign)")
    if rows:
        print("Next: python -m benchmark.build_missense_scores")
    else:
        print("No variants found — try --genes with richer ClinVar genes (e.g. TP53,BRCA1).")


if __name__ == "__main__":
    main()
