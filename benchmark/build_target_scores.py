"""Score the gene gold set with Open Targets + DepMap, merged with Elo.

Ranks the gene hypotheses with the LLM judge (Elo), scores each gene with
Open Targets (gene↔myeloma association) and DepMap (dependency), and writes
two merged files the Cross-check tab reads:
    benchmark/opentargets_scores.json   [{label, elo, score}, ...]
    benchmark/depmap_scores.json        [{label, elo, score}, ...]

Needs ANTHROPIC_API_KEY (for Elo). Open Targets is a free API (no key). DepMap
needs the public CRISPRGeneEffect.csv in data/depmap/ (or set DEPMAP_CSV); genes
missing a score are skipped per file.

    python -m benchmark.build_target_scores
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from co_scientist.state import Hypothesis
from benchmark.gold_set_genes import GOLD_GENES, DISEASE
from co_scientist.target_scorer import opentargets_score, depmap_score

HERE = Path(__file__).resolve().parent


def _elo_by_label(cycles: int, n_per_cycle: int) -> dict[str, float]:
    from benchmark.run_benchmark import run_tournament, _fair_judge_fn
    hyps, label_of = [], {}
    for g in GOLD_GENES:
        h = Hypothesis.new(statement=g["statement"], rationale=g["rationale"],
                           experiment="")
        label_of[h.id] = g["label"]
        hyps.append(h)
    run_tournament(hyps, _fair_judge_fn(ontology=False), cycles, n_per_cycle)
    return {label_of[h.id]: round(h.elo, 1) for h in hyps}


def _write(name: str, scorer, elo: dict) -> None:
    rows = []
    for g in GOLD_GENES:
        s = scorer(g["symbol"])
        if s is None:
            print(f"  {g['symbol']:8} ({name}): no score — skipping")
            continue
        rows.append({"label": g["label"], "elo": elo[g["label"]], "score": float(s)})
        print(f"  {g['symbol']:8} ({name}): {s:.4f}")
    out = HERE / f"{name}_scores.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"  -> wrote {out.name} ({len(rows)} rows)\n")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY — the judge ranks the genes for Elo.")
    print("Ranking gene hypotheses for Elo…")
    elo = _elo_by_label(6, 8)
    print(f"\nOpen Targets (gene ↔ {DISEASE} association):")
    _write("opentargets", lambda s: opentargets_score(s, DISEASE), elo)
    print("DepMap (gene dependency):")
    _write("depmap", depmap_score, elo)
    print("Now see the correlations in the Cross-check tab, or:")
    print("  python -m benchmark.elo_vs_variant_score --scores benchmark/opentargets_scores.json")
    print("  python -m benchmark.elo_vs_variant_score --scores benchmark/depmap_scores.json")


if __name__ == "__main__":
    main()
