"""LangGraph wiring of the Co-Scientist loop.

Supervisor -> [generation | reflection | ranking | evolution | meta_review]
            -> proximity (always after a generation/evolution change)
            -> back to supervisor

In production you'd run agents in parallel via LangGraph's `add_conditional_edges`
plus `Send`; this version is sequential for clarity.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from . import agents
from .state import CoScientistState


def build_graph(max_iterations: int = 12, use_fair_judge: bool = True):
    g = StateGraph(CoScientistState)

    # The Ranking node decides each pairwise match. The fair judge (default)
    # judges every pair in BOTH orders and only awards a win when the two
    # passes agree — neutralising position bias and scoring flips as draws.
    # Set use_fair_judge=False to fall back to the original single-pass judge.
    if use_fair_judge:
        from benchmark.fair_judge import ranking_fair
        ranking_node = ranking_fair
    else:
        ranking_node = agents.ranking

    g.add_node("supervisor", agents.supervisor)
    g.add_node("generation", agents.generation)
    g.add_node("reflection", agents.reflection)
    g.add_node("ranking", ranking_node)
    g.add_node("evolution", agents.evolution)
    g.add_node("meta_review", agents.meta_review)
    g.add_node("proximity", agents.proximity)

    g.set_entry_point("supervisor")

    def route(state: CoScientistState):
        if state.get("iteration", 0) >= max_iterations:
            return END
        return state.get("next_action", "generation")

    g.add_conditional_edges("supervisor", route, {
        "generation": "generation",
        "reflection": "reflection",
        "ranking": "ranking",
        "evolution": "evolution",
        "meta_review": "meta_review",
        END: END,
    })

    # After any agent finishes, run proximity, then supervisor decides again.
    for node in ("generation", "reflection", "ranking",
                 "evolution", "meta_review"):
        g.add_edge(node, "proximity")
    g.add_edge("proximity", "supervisor")

    return g.compile()
