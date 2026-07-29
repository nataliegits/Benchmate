"""Condense the DepMap CRISPR matrix into a file small enough to ship.

The problem: CRISPRGeneEffect.csv is ~440 MB — too big for git, so it's
gitignored, so DepMap silently doesn't exist on a hosted deploy. Telling every
user to go download it defeats the point of a web app.

The fix: the panel only ever needs one number per gene (mean gene effect). That's
~18,000 rows and about 0.5 MB, which commits happily. Run this once locally where
the full matrix lives; the summary it writes is what the hosted app reads.

    python -m benchmark.build_depmap_summary

If data/depmap/Model.csv is also present, the mean is restricted to the lineage in
DEPMAP_LINEAGE (default multiple myeloma) — a more on-target number than the
pan-cancer average. The lineage used is recorded in the file so nobody has to
guess later which one they're looking at.
"""
from __future__ import annotations

import sys

import pandas as pd

from co_scientist.target_scorer import (
    DEPMAP_CSV, DEPMAP_LINEAGE, DEPMAP_MODEL_CSV, _myeloma_model_ids)

OUT = DEPMAP_CSV.parent / "gene_effect_summary.csv"


def main() -> int:
    if not DEPMAP_CSV.exists():
        print(f"No full matrix at {DEPMAP_CSV}.\n"
              f"Download CRISPRGeneEffect.csv from "
              f"https://depmap.org/portal/data_page/ into {DEPMAP_CSV.parent}/ "
              f"and run this again.")
        return 1

    print(f"reading {DEPMAP_CSV.name} ({DEPMAP_CSV.stat().st_size / 1e6:.0f} MB)…")
    df = pd.read_csv(DEPMAP_CSV, index_col=0)
    print(f"  {df.shape[0]} cell lines x {df.shape[1]} genes")

    ids = _myeloma_model_ids()
    if ids:
        subset = df[df.index.isin(ids)]
        if len(subset) >= 3:
            df = subset
            lineage = DEPMAP_LINEAGE
            print(f"  restricted to {len(df)} {lineage} lines "
                  f"(Model.csv present)")
        else:
            lineage = "all"
            print(f"  only {len(subset)} {DEPMAP_LINEAGE} lines matched — too few "
                  f"to average, using all lines")
    else:
        lineage = "all"
        print(f"  Model.csv not found at {DEPMAP_MODEL_CSV.name} — using all "
              f"cancer lines (pan-cancer average)")

    means = df.mean(axis=0, skipna=True)
    counts = df.notna().sum(axis=0)

    # columns look like "SYVN1 (ENSG00000162298)" — split symbol from id so the
    # lookup doesn't have to re-parse strings on every call
    out = pd.DataFrame({
        "gene": [str(c).split(" ")[0].upper() for c in means.index],
        # store the raw mean effect (negative = essential). The sign flip to
        # "higher = more essential" belongs in the scorer, not the data file.
        "mean_effect": means.round(4).values,
        "n_lines": counts.values,
    })
    out = out[out["gene"] != ""].drop_duplicates(subset="gene")
    out["lineage"] = lineage

    out.to_csv(OUT, index=False)
    size_kb = OUT.stat().st_size / 1024
    print(f"\nwrote {OUT} — {len(out)} genes, {size_kb:.0f} KB, lineage={lineage}")
    print("Commit this file: it's what makes DepMap work on the hosted app.")

    ess = out.nsmallest(5, "mean_effect")
    print("\nsanity check — most essential genes (should be ribosomal / "
          "proteasome / polymerase):")
    for _, r in ess.iterrows():
        print(f"  {r['gene']:12} {r['mean_effect']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
