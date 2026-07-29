"""Option B: fetch REAL regulatory-variant coordinates from GTEx.

The variant gold set ships with placeholder coordinates, so AlphaGenome predicts
~no effect (all scores ≈ 0). This script replaces them with genuine eQTLs — DNA
variants empirically associated with each gene's expression — from the GTEx
Portal API (hg38 / gtex_v8, the same build AlphaGenome uses). Nothing is
invented: every coordinate comes back from GTEx.

For each ERAD gene it resolves the Ensembl/GENCODE id, pulls significant
single-tissue eQTLs, and picks the most significant one. It writes
`benchmark/real_variant_coords.json` (which gold_set_variants.py auto-loads) and
prints a Colab-ready VARIANTS block to paste into the scoring notebook.

    python -m benchmark.fetch_eqtls

Needs network access; uses httpx (already a dependency). If GTEx's API shape has
changed, the per-gene errors print the raw response so you can adjust.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from benchmark.gold_set_variants import GOLD_VARIANTS

GTEX = "https://gtexportal.org/api/v2"
DATASET = "gtex_v8"
OUT = Path(__file__).resolve().parent / "real_variant_coords.json"


def _gencode_id(client: httpx.Client, symbol: str) -> str | None:
    r = client.get(f"{GTEX}/reference/gene", params={"geneId": symbol})
    r.raise_for_status()
    data = r.json().get("data", [])
    for g in data:                       # prefer an exact symbol match
        if str(g.get("geneSymbol", "")).upper() == symbol.upper():
            return g.get("gencodeId")
    return data[0].get("gencodeId") if data else None


def _top_eqtl(client: httpx.Client, gencode_id: str) -> dict | None:
    r = client.get(f"{GTEX}/association/singleTissueEqtl",
                   params={"gencodeId": gencode_id, "datasetId": DATASET,
                           "itemsPerPage": 250})
    r.raise_for_status()
    rows = r.json().get("data", [])
    if not rows:
        return None
    return min(rows, key=lambda x: float(x.get("pValue", 1.0)))


def _parse_variant(variant_id: str) -> dict | None:
    # format: chr14_81000000_C_T_b38
    parts = variant_id.split("_")
    if len(parts) < 4:
        return None
    return {"chrom": parts[0], "pos": int(parts[1]), "ref": parts[2], "alt": parts[3]}


def fetch_for_genes(symbols, timeout: float = 30.0) -> tuple[dict, list[str]]:
    """Top significant eQTL per gene symbol, straight from GTEx.

    Returns ({SYMBOL: {chrom, pos, ref, alt, rsid, tissue, pvalue}}, problems).

    This is the gene-agnostic version of main(): it takes whatever genes the
    user's question is actually about instead of the fixed ERAD gold set. Genes
    with no significant eQTL are reported in `problems` and simply left out —
    the alternative would be inventing a coordinate, and a fabricated locus
    scores as "no effect", which is indistinguishable from a real null result.
    """
    coords: dict[str, dict] = {}
    problems: list[str] = []
    with httpx.Client(timeout=timeout,
                      headers={"Accept": "application/json"}) as client:
        for sym in symbols:
            sym = str(sym).strip().upper()
            if not sym or sym in coords:
                continue
            try:
                gid = _gencode_id(client, sym)
                if not gid:
                    problems.append(f"{sym}: not found in GTEx")
                    continue
                eqtl = _top_eqtl(client, gid)
                if not eqtl:
                    problems.append(f"{sym}: no significant eQTL in GTEx "
                                    f"(nothing for AlphaGenome to score)")
                    continue
                v = _parse_variant(eqtl.get("variantId", ""))
                if not v:
                    problems.append(f"{sym}: unparseable variant id "
                                    f"{eqtl.get('variantId')!r}")
                    continue
                v.update(rsid=eqtl.get("snpId"),
                         tissue=eqtl.get("tissueSiteDetailId"),
                         pvalue=eqtl.get("pValue"))
                coords[sym] = v
            except Exception as e:
                problems.append(f"{sym}: GTEx error — {e}")
    return coords, problems


def main():
    coords = {}
    with httpx.Client(timeout=30.0, headers={"Accept": "application/json"}) as client:
        for g in GOLD_VARIANTS:
            sym, label = g["gene"], g["label"]
            try:
                gid = _gencode_id(client, sym)
                if not gid:
                    print(f"  {sym:8} ({label}): no GENCODE id found"); continue
                eqtl = _top_eqtl(client, gid)
                if not eqtl:
                    print(f"  {sym:8} ({label}): no eQTLs returned"); continue
                v = _parse_variant(eqtl.get("variantId", ""))
                if not v:
                    print(f"  {sym:8} ({label}): could not parse "
                          f"{eqtl.get('variantId')!r}"); continue
                coords[label] = v
                print(f"  {sym:8} ({label}): {eqtl.get('variantId')}  "
                      f"rs={eqtl.get('snpId')}  tissue={eqtl.get('tissueSiteDetailId')}  "
                      f"p={eqtl.get('pValue')}")
            except Exception as e:
                print(f"  {sym:8} ({label}): error — {e}")

    if coords:
        OUT.write_text(json.dumps(coords, indent=2))
        print(f"\nWrote {OUT.name} with {len(coords)} real coordinates "
              "(gold_set_variants.py will auto-load it).\n")
        print("Colab VARIANTS block — paste into the scoring notebook (cell 3):\n")
        print("VARIANTS = [")
        for g in GOLD_VARIANTS:
            v = coords.get(g["label"])
            if v:
                print(f"    dict(label={g['label']!r}, chrom={v['chrom']!r}, "
                      f"pos={v['pos']}, ref={v['ref']!r}, alt={v['alt']!r}),")
        print("]")
    else:
        print("\nNo coordinates fetched — check network / the GTEx API shape above.")


if __name__ == "__main__":
    main()
