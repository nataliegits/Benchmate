"""A fairer pairwise judge — the highest-leverage fix to the tournament.

The problem with the judge in `agents.ranking`
----------------------------------------------
1. **Position bias.** Hypotheses are always shown in a fixed "A:" / "B:"
   order. LLM judges have a well-documented tendency to favour whichever
   option is shown first (or second). Your scheduler lists the higher-Elo
   hypothesis first, so position bias quietly *entrenches the current leader*
   and makes the ranking self-fulfilling instead of accurate.
2. **Thin evidence.** The judge only sees `statement` + `rationale`. The
   `experiment` (is it actually falsifiable?) and the reviewer notes never
   reach it.

The fix
-------
Judge each pair TWICE with the order swapped, and only count a win when both
passes agree on the *same hypothesis*. If they flip with the order, that's
position bias — we record it and score the match as a draw instead of letting
noise move the ratings. The prompt also includes the experiment and the
latest reviewer note, and asks for a per-criterion call.

Drop-in: see `ranking_fair` at the bottom — it's a direct replacement for
`co_scientist.agents.ranking`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from co_scientist.elo import schedule_matches, update_elo
from co_scientist.llm import call_json
from co_scientist.state import CoScientistState, Hypothesis


def _block(h: Hypothesis, exp_chars: int = 600) -> str:
    """Format a hypothesis for the judge — richer than statement+rationale."""
    note = h.review_notes[-1] if h.review_notes else ""
    parts = [f"Statement: {h.statement}",
             f"Rationale: {h.rationale}",
             f"Proposed experiment: {h.experiment[:exp_chars]}"]
    if note:
        parts.append(f"Prior reviewer note: {note}")
    return "\n".join(parts)


@dataclass
class Verdict:
    winner: str               # "A" | "B" | "draw"  (in terms of the ORIGINAL a, b)
    agreed: bool              # did the two order-swapped passes agree?
    position_biased: bool     # passes flipped with order -> bias / true toss-up
    reasons: tuple[str, str] = field(default_factory=lambda: ("", ""))


def _ask(first: Hypothesis, second: Hypothesis, crit_str: str,
         addendum: str, role: str, model: str | None) -> str:
    """One judgment. Returns 'first' or 'second' or 'draw' relative to the
    order the two hypotheses were passed in here."""
    out = call_json(
        "Two competing scientific hypotheses. Decide which is stronger overall, "
        f"weighing these criteria equally: {crit_str}.\n\n"
        f"--- HYPOTHESIS X ---\n{_block(first)}\n\n"
        f"--- HYPOTHESIS Y ---\n{_block(second)}\n\n"
        "Judge on scientific merit only; ignore length and ordering. "
        'Output JSON: {"winner": "X" | "Y" | "draw", '
        '"reason": "one sentence under 35 words"}'
        + addendum,
        system="You are a rigorous, impartial scientific reviewer.",
        role=role, model=model, temperature=0.2, max_tokens=400,
    )
    w = str(out.get("winner", "draw")).strip().upper()
    return {"X": "first", "Y": "second"}.get(w, "draw"), out.get("reason", "")


def judge_pair(a: Hypothesis, b: Hypothesis, criteria: list[str],
               *, addendum: str = "", role: str = "ranking",
               model: str | None = None) -> Verdict:
    """Order-swapped, bias-aware pairwise judgment.

    Pass 1 shows (a, b); pass 2 shows (b, a). Translate both back to a/b and
    compare. Agreement -> confident winner. Disagreement -> position bias,
    scored as a draw.
    """
    crit_str = ", ".join(criteria)
    r1, reason1 = _ask(a, b, crit_str, addendum, role, model)        # a is "X"
    r2, reason2 = _ask(b, a, crit_str, addendum, role, model)        # b is "X"

    # Map each pass to the winning ORIGINAL hypothesis ("A"=a, "B"=b)
    pass1 = {"first": "A", "second": "B", "draw": "draw"}[r1]
    pass2 = {"first": "B", "second": "A", "draw": "draw"}[r2]

    if pass1 == pass2:
        return Verdict(pass1, agreed=True, position_biased=False,
                       reasons=(reason1, reason2))
    # One side called a draw, the other a winner -> weak signal, treat as draw
    if "draw" in (pass1, pass2):
        return Verdict("draw", agreed=False, position_biased=False,
                       reasons=(reason1, reason2))
    # Clear flip with order -> position bias
    return Verdict("draw", agreed=False, position_biased=True,
                   reasons=(reason1, reason2))


# ----- drop-in replacement for co_scientist.agents.ranking ----------------

def ranking_fair(state: CoScientistState) -> dict:
    """Same contract as agents.ranking, but uses the order-swapped judge and
    reports how often position bias showed up.

    To use it, in co_scientist/graph.py swap the ranking node:
        from benchmark.fair_judge import ranking_fair
        g.add_node("ranking", ranking_fair)
    (Costs ~2x judge calls per match. Combine with a higher n_matches budget —
    see BENCHMARKING_PLAN.md — and keep Haiku as the ranking model.)
    """
    hypotheses = state["hypotheses"]
    matches = schedule_matches(hypotheses, n_matches=state.get("n_matches", 8))
    criteria = state["plan_config"].get(
        "evaluation_criteria", ["novelty", "plausibility", "testability"])
    crit = state.get("meta_critique")
    addendum = crit.as_prompt_addendum() if crit else ""

    biased = 0
    for a, b in matches:
        v = judge_pair(a, b, criteria, addendum=addendum, role="ranking")
        biased += int(v.position_biased)
        if v.winner == "A":
            update_elo(a, b)
        elif v.winner == "B":
            update_elo(b, a)
        else:
            update_elo(a, b, draw=True)
    if matches:
        print(f"[ranking_fair] {biased}/{len(matches)} matches flagged "
              f"position-biased (scored as draws)")
    return {"hypotheses": hypotheses}
