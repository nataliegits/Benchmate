"""The seven Co-Scientist agents, implemented as LangGraph nodes.

Each node takes the shared state, does its thing, and returns updates.
LangGraph merges them back into state.
"""
from __future__ import annotations

import random
import re
from typing import Any

from .elo import schedule_matches, update_elo
from .llm import call, call_json
from .state import (
    CoScientistState, Hypothesis, MetaCritique, INITIAL_ELO,
)
from .tools import (
    pubmed_search, embed, cosine,
    available_geneformer_genes, geneformer_neighbors,
)


SYSTEM_BASE = (
    "You are a domain expert biomedical scientist participating in a "
    "structured hypothesis-generation system. Be precise, cite evidence "
    "when asked, and flag uncertainty explicitly."
)


def _addendum(state: CoScientistState) -> str:
    """Inject Meta-review feedback into every agent's prompt — this is the
    'learning without backprop' mechanism from the paper."""
    crit = state.get("meta_critique")
    return crit.as_prompt_addendum() if crit else ""


def _geneformer_context_for(text: str, top_n: int = 10) -> str:
    """Find genes mentioned in `text` that we have cached perturbation data for,
    and format their top affected genes as a prompt-ready evidence block.

    Returns "" if no relevant genes are mentioned or no data is available.
    """
    available = available_geneformer_genes()
    if not available:
        return ""
    mentioned = [g for g in available if re.search(rf"\b{re.escape(g)}\b", text)]
    if not mentioned:
        return ""
    blocks = []
    for sym in mentioned:
        r = geneformer_neighbors(sym, top_n=top_n)
        if "error" in r or not r.get("affected_genes"):
            continue
        lines = [
            f"  {ag['symbol']:>10}  Δcos={ag['cosine_shift']:.3f}  N={ag['n_detections']}"
            for ag in r["affected_genes"]
        ]
        blocks.append(
            f"In-silico KO of {sym} — top {len(lines)} affected genes "
            f"(Δcos = predicted embedding shift, larger = bigger effect):\n"
            + "\n".join(lines)
        )
    return "\n\n".join(blocks)


# ============================================================
# 1. Supervisor — parses goal, picks next action
# ============================================================

def supervisor(state: CoScientistState) -> dict[str, Any]:
    if "plan_config" not in state:
        plan = call_json(
            f"Research goal: {state['research_goal']}\n\n"
            "Extract a research-plan configuration. Output JSON with keys:\n"
            "  evaluation_criteria: list of 3-5 criteria for judging hypotheses\n"
            "  constraints:        any explicit constraints from the goal\n"
            "  initial_search_terms: 3-5 PubMed search queries to seed the work",
            system=SYSTEM_BASE, role="supervisor",
        )
        return {"plan_config": plan, "iteration": 0,
                "hypotheses": [], "meta_critique": MetaCritique(),
                "next_action": "generation"}

    # Decide what to do next based on system state.
    n = len(state.get("hypotheses", []))
    it = state.get("iteration", 0)

    if n < 6:
        action = "generation"
    elif it % 4 == 1:
        action = "reflection"
    elif it % 4 == 2:
        action = "ranking"
    elif it % 4 == 3:
        action = "evolution"
    else:
        action = "meta_review"

    return {"next_action": action, "iteration": it + 1}


# ============================================================
# 2. Generation — proposes new hypotheses
# ============================================================

def generation(state: CoScientistState) -> dict[str, Any]:
    goal = state["research_goal"]
    plan = state["plan_config"]

    # Pull literature context.
    queries = plan.get("initial_search_terms", [goal])[:3]
    lit_blocks = []
    for q in queries:
        try:
            papers = pubmed_search(q, max_results=3)
            lit_blocks.extend(p.short() for p in papers)
        except Exception as e:
            lit_blocks.append(f"(PubMed error for '{q}': {e})")
    lit = "\n\n".join(lit_blocks) or "(no literature retrieved)"

    # Geneformer evidence if the goal mentions any gene we have data for.
    gf = _geneformer_context_for(goal, top_n=10)
    gf_block = (f"\n\nCached Geneformer in-silico perturbation evidence "
                f"(use to ground hypotheses in real perturbation data):\n{gf}\n"
                if gf else "")

    existing = state.get("hypotheses", [])
    existing_summary = "\n".join(
        f"- {h.statement[:120]}" for h in sorted(existing, key=lambda h: -h.elo)[:5]
    ) or "(none yet)"

    out = call_json(
        f"Research goal: {goal}\n\n"
        f"Relevant literature:\n{lit}"
        f"{gf_block}\n"
        f"Existing top hypotheses (do NOT duplicate):\n{existing_summary}\n\n"
        "Propose 3 NOVEL, testable hypotheses that address the goal. For each, "
        "output an object with keys: statement, rationale, experiment, citations "
        "(list of PMIDs from the literature above, if applicable). Return a JSON "
        "object: {\"hypotheses\": [..., ..., ...]}."
        + _addendum(state),
        system=SYSTEM_BASE, role="generation",
        temperature=0.9, max_tokens=3000,
    )

    new = [Hypothesis.new(**h) for h in out["hypotheses"]]
    return {"hypotheses": existing + new}


# ============================================================
# 3. Reflection — critically reviews hypotheses
# ============================================================

def reflection(state: CoScientistState) -> dict[str, Any]:
    hypotheses = state["hypotheses"]
    # Review the most under-reviewed hypotheses first.
    targets = sorted(hypotheses, key=lambda h: len(h.review_notes))[:3]
    for h in targets:
        gf = _geneformer_context_for(
            f"{h.statement} {h.rationale} {h.experiment}", top_n=8)
        gf_block = (f"\n\nGeneformer perturbation evidence for genes "
                    f"mentioned above:\n{gf}\n"
                    f"Use this to check whether the hypothesis's mechanism "
                    f"is consistent with the predicted affected genes."
                    if gf else "")
        critique = call(
            f"Critically review this hypothesis as a peer reviewer.\n\n"
            f"Statement: {h.statement}\n"
            f"Rationale: {h.rationale}\n"
            f"Proposed experiment: {h.experiment}"
            f"{gf_block}\n\n"
            "In 4 sentences max, identify: (a) the strongest objection, "
            "(b) whether the experiment as designed could falsify it, "
            "(c) one concrete improvement."
            + _addendum(state),
            system=SYSTEM_BASE, role="reflection",
            temperature=0.4, max_tokens=400,
        )
        h.review_notes.append(critique.strip())
    return {"hypotheses": hypotheses}


# ============================================================
# 4. Ranking — Elo tournament via LLM-as-judge scientific debate
# ============================================================

def ranking(state: CoScientistState) -> dict[str, Any]:
    hypotheses = state["hypotheses"]
    matches = schedule_matches(hypotheses, n_matches=state.get("n_matches", 8))
    criteria = state["plan_config"].get("evaluation_criteria",
                                        ["novelty", "plausibility", "testability"])
    crit_str = ", ".join(criteria)

    for a, b in matches:
        verdict = call_json(
            f"Two competing hypotheses. Judge which is better.\n\n"
            f"A: {a.statement}\n   rationale: {a.rationale}\n\n"
            f"B: {b.statement}\n   rationale: {b.rationale}\n\n"
            f"Evaluation criteria: {crit_str}\n\n"
            "Output JSON: {\"winner\": \"A\" | \"B\" | \"draw\", "
            "\"reason\": \"one sentence under 40 words\"}"
            + _addendum(state),
            system=SYSTEM_BASE, role="ranking",
            temperature=0.2, max_tokens=500,
        )
        if verdict["winner"] == "A":
            update_elo(a, b)
        elif verdict["winner"] == "B":
            update_elo(b, a)
        else:
            update_elo(a, b, draw=True)

    return {"hypotheses": hypotheses}


# ============================================================
# 5. Proximity — clusters similar hypotheses (de-dup hint for Ranking)
# ============================================================

def proximity(state: CoScientistState) -> dict[str, Any]:
    """No state mutation — this would normally build a graph used by Ranking
    to schedule informative pairings. Here we just log near-duplicates."""
    hyps = state["hypotheses"]
    embeds = [embed(h.statement) for h in hyps]
    dupes: list[tuple[str, str, float]] = []
    for i in range(len(hyps)):
        for j in range(i + 1, len(hyps)):
            sim = cosine(embeds[i], embeds[j])
            if sim > 0.85:
                dupes.append((hyps[i].id, hyps[j].id, sim))
    if dupes:
        print(f"[proximity] {len(dupes)} near-duplicate pairs detected")
    return {}


# ============================================================
# 6. Evolution — refines the top hypotheses into new variants
# ============================================================

def evolution(state: CoScientistState) -> dict[str, Any]:
    hyps = state["hypotheses"]
    if not hyps:
        return {}
    top = sorted(hyps, key=lambda h: -h.elo)[:3]

    strategies = [
        "Sharpen this hypothesis by making the proposed experiment more falsifiable.",
        "Combine the strongest elements of this with another top hypothesis "
        "below it in the ranking.",
        "Generate a more parsimonious version that makes a stronger commitment.",
    ]
    new_hyps: list[Hypothesis] = []
    for h, strategy in zip(top, strategies):
        out = call_json(
            f"Improve this hypothesis. Strategy: {strategy}\n\n"
            f"Original statement: {h.statement}\n"
            f"Original rationale: {h.rationale}\n"
            f"Original experiment: {h.experiment}\n"
            f"Reviewer critiques: {' | '.join(h.review_notes[-2:]) or 'none'}\n\n"
            "Output a JSON object with keys: statement, rationale, experiment. "
            "Keep the experiment field under 1500 words so the JSON closes."
            + _addendum(state),
            system=SYSTEM_BASE, role="evolution",
            temperature=0.6, max_tokens=4000,
        )
        new = Hypothesis.new(
            statement=out["statement"], rationale=out["rationale"],
            experiment=out["experiment"], parent_ids=[h.id],
            generation=h.generation + 1,
        )
        new_hyps.append(new)
    return {"hypotheses": hyps + new_hyps}


# ============================================================
# 7. Meta-review — synthesises patterns into prompt-level feedback
# ============================================================

def meta_review(state: CoScientistState) -> dict[str, Any]:
    hyps = state["hypotheses"]
    if not hyps:
        return {}

    top = sorted(hyps, key=lambda h: -h.elo)[:5]
    bottom = sorted(hyps, key=lambda h: h.elo)[:5]

    sample = "TOP-RANKED HYPOTHESES:\n" + "\n".join(
        f"- ({h.elo:.0f}) {h.statement[:200]}" for h in top
    ) + "\n\nBOTTOM-RANKED HYPOTHESES:\n" + "\n".join(
        f"- ({h.elo:.0f}) {h.statement[:200]}" for h in bottom
    )

    out = call_json(
        f"You are conducting a meta-review across this tournament's results.\n\n"
        f"{sample}\n\nSelected reviewer critiques across hypotheses:\n"
        + "\n".join(f"- {n}" for h in hyps for n in h.review_notes[:1])
        + "\n\nIdentify recurring issues that pulled bottom hypotheses down, "
        "and successful patterns that lifted top hypotheses up. "
        "Output JSON: {\"recurring_issues\": [str, ...], "
        "\"successful_patterns\": [str, ...]} with at most 4 of each, "
        "phrased as actionable rules.",
        system=SYSTEM_BASE, role="meta_review",
        temperature=0.3, max_tokens=900,
    )
    return {"meta_critique": MetaCritique(
        recurring_issues=out.get("recurring_issues", []),
        successful_patterns=out.get("successful_patterns", []),
    )}
