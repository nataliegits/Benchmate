"""Shared state for the Co-Scientist loop.

The TypedDict at the bottom is the LangGraph state. Everything else is the
data model — keep it boring on purpose so it serialises to JSON cleanly.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, TypedDict

INITIAL_ELO = 1200


@dataclass
class Hypothesis:
    id: str
    statement: str            # one-paragraph hypothesis
    rationale: str            # why this is plausible
    experiment: str           # how you'd test it
    citations: list[str] = field(default_factory=list)
    elo: float = INITIAL_ELO
    matches_played: int = 0
    review_notes: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)   # for Evolution lineage
    generation: int = 0       # how many evolution rounds in

    @classmethod
    def new(cls, statement: str, rationale: str, experiment: str, **kw) -> "Hypothesis":
        return cls(id=uuid.uuid4().hex[:8], statement=statement,
                   rationale=rationale, experiment=experiment, **kw)


@dataclass
class MetaCritique:
    """Feedback the Meta-review agent writes back into every other agent's prompt."""
    recurring_issues: list[str] = field(default_factory=list)
    successful_patterns: list[str] = field(default_factory=list)

    def as_prompt_addendum(self) -> str:
        if not self.recurring_issues and not self.successful_patterns:
            return ""
        parts = ["\n--- Lessons from previous rounds ---"]
        if self.recurring_issues:
            parts.append("Common mistakes to avoid:")
            parts.extend(f"- {x}" for x in self.recurring_issues)
        if self.successful_patterns:
            parts.append("Patterns that have worked well:")
            parts.extend(f"- {x}" for x in self.successful_patterns)
        return "\n".join(parts)


class CoScientistState(TypedDict, total=False):
    """LangGraph state. Mutated in-place by agent nodes."""
    research_goal: str
    plan_config: dict[str, Any]       # parsed criteria, constraints
    hypotheses: list[Hypothesis]
    iteration: int
    max_iterations: int
    n_matches: int                    # tournament matches per ranking round
    meta_critique: MetaCritique
    next_action: str                  # which agent to run next


# ----------------- JSON persistence -----------------

def to_jsonable(state: CoScientistState) -> dict:
    s = dict(state)
    s["hypotheses"] = [asdict(h) for h in state.get("hypotheses", [])]
    if "meta_critique" in state:
        s["meta_critique"] = asdict(state["meta_critique"])
    return s


def from_jsonable(d: dict) -> CoScientistState:
    s: CoScientistState = dict(d)  # type: ignore
    s["hypotheses"] = [Hypothesis(**h) for h in d.get("hypotheses", [])]
    if "meta_critique" in d:
        s["meta_critique"] = MetaCritique(**d["meta_critique"])
    return s


def save(state: CoScientistState, path: str | Path = "state.json") -> None:
    Path(path).write_text(json.dumps(to_jsonable(state), indent=2))


def load(path: str | Path = "state.json") -> CoScientistState | None:
    p = Path(path)
    if not p.exists():
        return None
    return from_jsonable(json.loads(p.read_text()))
