"""Programmatic Benchmate API for Hermes (or any external orchestrator).

This module exposes three functions that compose into a full research run:

  list_cache()                  → what genes already have perturbation data
  add_perturbation(syms, cell)  → generate a Colab notebook for new genes
  run_benchmate(goal, n)        → execute the agent loop, return top hypotheses

Each function is JSON-serializable in/out so Hermes can wire them into
its skill system without thinking about Python types.

Usage from Hermes (after registering this module as a skill):

    skill: benchmate_run
    description: Generate hypotheses about a biomedical research goal
    handler: hermes.benchmate_runner.run_benchmate
    parameters:
      goal: str
      max_iterations: int (default 8)
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

# Make repo imports work whether hermes/ is on sys.path or not
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from co_scientist import state as st
from co_scientist.graph import build_graph
from co_scientist.tools import available_geneformer_genes, geneformer_neighbors


# ────────────────────────────────────────────────────────────
# 1. Cache introspection
# ────────────────────────────────────────────────────────────

def list_cache() -> dict[str, Any]:
    """Which genes have cached Geneformer perturbation results?

    Returns:
        {"cached_genes": ["TXNDC15", "SYVN1", "MARCHF6"]}
    """
    return {"cached_genes": available_geneformer_genes()}


def gene_neighbors(gene_symbol: str, top_n: int = 10) -> dict[str, Any]:
    """Top affected genes when `gene_symbol` is in-silico deleted.

    Wraps the same function the Generation agent uses internally, so
    Hermes can answer "what does TXNDC15 knockout perturb?" directly.
    """
    return geneformer_neighbors(gene_symbol, top_n=top_n)


# ────────────────────────────────────────────────────────────
# 2. Generate a new Colab notebook (for genes not yet in cache)
# ────────────────────────────────────────────────────────────

def add_perturbation(symbols: list[str],
                     cell_context: str = "Plasma cells (extreme ER stress)"
                     ) -> dict[str, Any]:
    """Generate a parameterised Colab notebook for `symbols` in `cell_context`.

    Returns the path to the .ipynb plus the mygene-resolved Ensembl IDs.
    Hermes can then post the notebook URL to whatever chat platform.

    Note: this does NOT run the perturbation (Geneformer needs a GPU).
    Hermes should either:
      - hand the notebook to a human ("here's the link, run it"), OR
      - use its Modal sandbox backend to run the notebook automatically.
    """
    from ui.notebook_gen import generate_notebook, CELL_TYPE_PRESETS
    if cell_context not in CELL_TYPE_PRESETS:
        return {"error": f"Unknown cell_context '{cell_context}'. "
                         f"Available: {list(CELL_TYPE_PRESETS)}"}
    try:
        nb_path, resolved = generate_notebook(symbols, preset_name=cell_context)
    except Exception as e:
        return {"error": str(e)}
    return {
        "notebook_path": str(nb_path),
        "resolved": resolved,
        "cell_context": cell_context,
        "next_step": ("Open this notebook in Colab, Run All on a T4 GPU "
                      "(~1h), then drop the *_stats.csv files into "
                      "data/geneformer/."),
    }


# ────────────────────────────────────────────────────────────
# 3. Run the Benchmate agent loop
# ────────────────────────────────────────────────────────────

def run_benchmate(goal: str, max_iterations: int = 8,
                  capture_logs: bool = True) -> dict[str, Any]:
    """Run the full Co-Scientist loop and return top hypotheses.

    Args:
        goal: research goal in natural language
        max_iterations: cap on supervisor cycles (default 8 ≈ $1-3 spend)
        capture_logs: if True, capture stdout into the response

    Returns:
        {
          "iterations_run": int,
          "n_hypotheses": int,
          "top_hypotheses": [
              {"elo": float, "statement": str, "rationale": str,
               "experiment": str, "matches_played": int, "generation": int},
              ... up to 5
          ],
          "logs": str (only if capture_logs)
        }
    """
    initial_state = {"research_goal": goal, "max_iterations": max_iterations}
    graph = build_graph(max_iterations=max_iterations)

    buf = io.StringIO()
    ctx = redirect_stdout(buf) if capture_logs else _passthrough()
    with ctx:
        final = graph.invoke(initial_state, {"recursion_limit": 200})

    hyps = sorted(final.get("hypotheses", []), key=lambda h: -h.elo)[:5]
    out: dict[str, Any] = {
        "iterations_run": final.get("iteration", 0),
        "n_hypotheses": len(final.get("hypotheses", [])),
        "top_hypotheses": [
            {
                "elo": round(h.elo, 1),
                "statement": h.statement,
                "rationale": h.rationale,
                "experiment": h.experiment,
                "matches_played": h.matches_played,
                "generation": h.generation,
            }
            for h in hyps
        ],
    }
    if capture_logs:
        out["logs"] = buf.getvalue()
    return out


class _passthrough:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ────────────────────────────────────────────────────────────
# CLI shim: call any of the above from `python -m hermes.benchmate_runner ...`
# Makes it dead simple to register as a Hermes shell command.
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Benchmate API for Hermes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-cache")

    p_neigh = sub.add_parser("neighbors")
    p_neigh.add_argument("gene")
    p_neigh.add_argument("--top-n", type=int, default=10)

    p_add = sub.add_parser("add-perturbation")
    p_add.add_argument("genes", nargs="+")
    p_add.add_argument("--cell-context", default="Plasma cells (extreme ER stress)")

    p_run = sub.add_parser("run")
    p_run.add_argument("goal")
    p_run.add_argument("--max-iterations", type=int, default=8)

    args = ap.parse_args()
    if args.cmd == "list-cache":
        result = list_cache()
    elif args.cmd == "neighbors":
        result = gene_neighbors(args.gene, top_n=args.top_n)
    elif args.cmd == "add-perturbation":
        result = add_perturbation(args.genes, cell_context=args.cell_context)
    elif args.cmd == "run":
        result = run_benchmate(args.goal, max_iterations=args.max_iterations)
    print(json.dumps(result, indent=2, default=str))
