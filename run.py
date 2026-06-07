"""Entry point. Run with:

    python run.py "Find novel drug repurposing candidates for AML"

Add --resume to continue from state.json.
Add --max-iterations N to cap the loop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from co_scientist import state as st
from co_scientist.graph import build_graph


console = Console()


def print_top(state, top_n: int = 5) -> None:
    # Sort: tested hypotheses first (by Elo), then untested at the bottom.
    # A 0-match hypothesis is still at the default 1200 Elo — it hasn't been
    # in any tournament round yet, so its rating is not a quality signal.
    all_hyps = state.get("hypotheses", [])
    hyps = sorted(
        all_hyps,
        key=lambda h: (h.matches_played > 0, h.elo),
        reverse=True,
    )[:top_n]

    table = Table(title=f"Top hypotheses (iter {state.get('iteration', 0)})",
                  show_lines=True)
    table.add_column("Elo", justify="right", style="bold")
    table.add_column("Matches", justify="right")
    table.add_column("Gen", justify="right")
    table.add_column("Statement", overflow="fold")
    for h in hyps:
        elo_label = (f"{h.elo:.0f}" if h.matches_played > 0
                     else f"{h.elo:.0f} (unranked)")
        table.add_row(elo_label, str(h.matches_played),
                      str(h.generation), h.statement)
    console.print(table)


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("goal", nargs="?",
                    help="Research goal in natural language")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from state.json if present")
    ap.add_argument("--max-iterations", type=int, default=12)
    ap.add_argument("--n-matches", type=int, default=8,
                    help="Tournament matches per ranking round. Higher = a more "
                         "stable leaderboard (aim for ~12 matches/hypothesis "
                         "across the whole run). See benchmark/BENCHMARKING_PLAN.md.")
    ap.add_argument("--naive-judge", action="store_true",
                    help="Use the original single-pass judge instead of the "
                         "order-swapped fair judge (faster, but position-biased).")
    ap.add_argument("--state-file", default="state.json")
    args = ap.parse_args()

    if args.resume and Path(args.state_file).exists():
        state = st.load(args.state_file)
        state["n_matches"] = args.n_matches      # let the CLI flag apply on resume
        console.print(f"[green]Resumed[/] from {args.state_file} "
                      f"(iter {state.get('iteration', 0)})")
    elif args.goal:
        state = {"research_goal": args.goal,
                 "max_iterations": args.max_iterations,
                 "n_matches": args.n_matches}
    else:
        ap.print_usage()
        return 1

    graph = build_graph(max_iterations=args.max_iterations,
                        use_fair_judge=not args.naive_judge)
    console.print(f"[dim]judge={'fair (order-swapped)' if not args.naive_judge else 'naive'}"
                  f" · {args.n_matches} matches/ranking round[/]")
    final = graph.invoke(state, {"recursion_limit": 200})

    print_top(final, top_n=5)
    st.save(final, args.state_file)
    console.print(f"\n[dim]State saved to {args.state_file}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
