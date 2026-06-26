"""Fetch REAL protein sequences from UniProt for the binding gold set.

`gold_set_binding.py` ships with placeholder sequences ("MELE_PLACEHOLDER_…")
that Boltz can't fold. This pulls the canonical reviewed (Swiss-Prot) human
sequence for each target from the UniProt REST API by gene symbol, writes
`benchmark/real_binding_seqs.json` ({label: sequence}) — which
`gold_set_binding.py` auto-loads — and prints the entry name + length so you can
sanity-check what it grabbed. Nothing is invented; every sequence comes from
UniProt.

    python -m benchmark.fetch_uniprot

Needs network; uses httpx (already a dependency).
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from benchmark.gold_set_binding import GOLD_BINDING

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
OUT = Path(__file__).resolve().parent / "real_binding_seqs.json"


def _fetch_fasta(client: httpx.Client, gene: str) -> tuple[str, str] | None:
    """Return (header, sequence) for the canonical reviewed human entry, or None."""
    query = f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true"
    r = client.get(UNIPROT, params={"query": query, "format": "fasta", "size": 1})
    r.raise_for_status()
    text = r.text.strip()
    if not text.startswith(">"):
        return None
    lines = text.splitlines()
    header = lines[0]
    seq = "".join(lines[1:])
    return (header, seq) if seq else None


def main():
    # gene symbol per label (multiple labels can share a target)
    by_label = {g["label"]: g.get("uniprot_gene") for g in GOLD_BINDING}
    seqs: dict[str, str] = {}
    seq_cache: dict[str, tuple[str, str]] = {}

    with httpx.Client(timeout=30.0, headers={"Accept": "text/plain"}) as client:
        for label, gene in by_label.items():
            if not gene:
                print(f"  {label}: no uniprot_gene set — skipping"); continue
            try:
                if gene not in seq_cache:
                    got = _fetch_fasta(client, gene)
                    if not got:
                        print(f"  {label} ({gene}): no reviewed human entry found")
                        continue
                    seq_cache[gene] = got
                header, seq = seq_cache[gene]
                seqs[label] = seq
                # header looks like: sp|P55072|TERA_HUMAN ... -> show accession + len
                acc = header.split("|")[1] if "|" in header else "?"
                print(f"  {label:18} {gene:7} -> {acc}  ({len(seq)} aa)  {header[:60]}")
            except Exception as e:
                print(f"  {label} ({gene}): error — {e}")

    if seqs:
        OUT.write_text(json.dumps(seqs, indent=2))
        print(f"\nWrote {OUT.name} with {len(seqs)} sequences "
              "(gold_set_binding.py will auto-load it).")
        print("Sanity-check the accessions above against what you expect "
              "(VCP=P55072, PSMB5=P28074, SYVN1=Q86TM6, EDEM1=Q92611).")
        print("\nNext: export BOLTZ_API_KEY + ANTHROPIC_API_KEY, then "
              "`python -m benchmark.build_boltz_scores`.")
    else:
        print("\nNo sequences fetched — check network / the UniProt API.")


if __name__ == "__main__":
    main()
