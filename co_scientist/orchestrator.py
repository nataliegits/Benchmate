"""The orchestrator — one agent everything reports back to.

Benchmate's capabilities used to be scattered across tabs and buttons. This puts
a single agent in front of them: it sees the project state, picks a tool, and
reports the result back into one conversation.

Two design rules:

  * **Tool-calling, not a menu.** Claude is given the real tool schemas and
    chooses (and can chain across turns). Falls back to a JSON planner if the
    provider doesn't return native tool calls.
  * **Human in the loop.** Every tool declares `writes` — anything that spends
    API credits or writes evidence is *proposed*, not executed. The UI shows a
    "waiting for you" card with a guidance box; the user can approve, amend, or
    skip. Read-only tools run immediately.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent

# NOTE: the LLM stack is imported lazily inside decide()/interpret(). Read-only
# tools (inventory lookups, leaderboard, bench results) then work with no model
# dependency at all — handy for tests and for a key-less session.


# ---------------------------------------------------------------------------
# Tool implementations (thin wrappers over the existing modules)
# ---------------------------------------------------------------------------

def _t_show_leaderboard(top_n: int = 5) -> dict:
    p = REPO / "state.json"
    if not p.exists():
        return {"error": "No run on record yet — run the tournament first."}
    hyps = sorted(json.loads(p.read_text()).get("hypotheses", []),
                  key=lambda h: (h.get("matches_played", 0) > 0, h.get("elo", 0)),
                  reverse=True)[:int(top_n)]
    return {"hypotheses": [{"elo": round(h.get("elo", 0)),
                            "matches": h.get("matches_played", 0),
                            "statement": h.get("statement", "")} for h in hyps]}


def _t_find_reagent(reagent: str) -> dict:
    from . import freezer
    box = freezer.load_box(freezer.DEFAULT_BOX)
    hits = freezer.locate(reagent, box)
    return {"reagent": reagent,
            "found": bool(hits),
            "position": hits[0]["position"] if hits else None,
            "label": hits[0]["label"] if hits else None}


def _t_check_reagents(reagents: list[str]) -> dict:
    from . import freezer
    box = freezer.load_box(freezer.DEFAULT_BOX)
    rows = freezer.reconcile(list(reagents), box)
    return {"checked": rows,
            "missing": [r["reagent"] for r in rows if not r["found"]]}


def _t_cross_check_summary() -> dict:
    out = {}
    for name in ("variant", "boltz", "opentargets", "depmap", "alphamissense"):
        p = REPO / "benchmark" / f"{name}_scores.json"
        if p.exists():
            try:
                rows = json.loads(p.read_text())
                out[name] = {"n": len(rows),
                             "top": max(rows, key=lambda r: r.get("score", 0)).get("label")
                             if rows else None}
            except Exception:
                pass
    return out or {"note": "No cross-check score files on record yet."}


def _t_bench_results() -> dict:
    from . import assay
    labels = assay.available_assays()
    recs = [assay.assay_evidence(l) for l in labels]
    return {"results": [{"label": r["hypothesis_label"], "drug": r["drug"],
                         "verdict": r["viability"]["verdict"],
                         "action": r["direction_for_benchmate"]}
                        for r in recs if r]} or {"results": []}


def _t_design_experiment(hypothesis: str) -> dict:
    from . import experiment
    return experiment.design_experiment(hypothesis)


def _t_refine_hypothesis(hypothesis: str, result_summary: str) -> dict:
    from . import experiment
    return experiment.refine_hypothesis(hypothesis, result_summary)


# ---------------------------------------------------------------------------
# Registry.  `writes=True` ⇒ costs money or writes evidence ⇒ needs approval.
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "show_leaderboard": dict(
        fn=_t_show_leaderboard, writes=False,
        desc="Show the current top-ranked hypotheses and their Elo from the last run.",
        params={"top_n": {"type": "integer", "description": "how many to show (default 5)"}},
        required=[]),
    "find_reagent": dict(
        fn=_t_find_reagent, writes=False,
        desc="Look up ONE reagent in the freezer inventory and return its slot.",
        params={"reagent": {"type": "string", "description": "reagent name, e.g. CB-5083"}},
        required=["reagent"]),
    "check_reagents": dict(
        fn=_t_check_reagents, writes=False,
        desc="Check a LIST of reagents against the inventory; returns which are missing.",
        params={"reagents": {"type": "array", "items": {"type": "string"},
                             "description": "reagent names"}},
        required=["reagents"]),
    "cross_check_summary": dict(
        fn=_t_cross_check_summary, writes=False,
        desc="Summarise which independent-model cross-check scores are on record.",
        params={}, required=[]),
    "bench_results": dict(
        fn=_t_bench_results, writes=False,
        desc="List wet-lab assay results on record and what each implies.",
        params={}, required=[]),
    "design_experiment": dict(
        fn=_t_design_experiment, writes=True,
        desc="Design a runnable alamarBlue experiment for a hypothesis (costs an LLM call).",
        params={"hypothesis": {"type": "string", "description": "the hypothesis to test"}},
        required=["hypothesis"]),
    "refine_hypothesis": dict(
        fn=_t_refine_hypothesis, writes=True,
        desc="Update a hypothesis in light of a bench result (costs an LLM call).",
        params={"hypothesis": {"type": "string"},
                "result_summary": {"type": "string",
                                   "description": "what the assay showed"}},
        required=["hypothesis", "result_summary"]),
}


def tool_schemas() -> list[dict]:
    return [{"type": "function",
             "function": {"name": name,
                          "description": t["desc"],
                          "parameters": {"type": "object",
                                         "properties": t["params"],
                                         "required": t["required"]}}}
            for name, t in TOOLS.items()]


SYSTEM = (
    "You are Benchmate, an AI co-scientist working alongside a bench scientist. "
    "You coordinate a set of tools: the hypothesis leaderboard, independent-model "
    "cross-checks, wet-lab results, the freezer inventory, experiment design, and "
    "hypothesis refinement.\n"
    "Be concise and collegial — a smart colleague, not a chatbot. When a tool "
    "result comes back, interpret it in one or two sentences rather than dumping "
    "it. Flag uncertainty plainly, and say when the evidence is thin (small n, a "
    "proxy assay, no control). Never invent data or mechanisms. If the user just "
    "wants to talk, answer without calling a tool."
)


def decide(user_message: str, history: list[dict], context: str = "") -> dict:
    """Pick a tool (or just reply). Returns {tool, args, say}."""
    msgs = [{"role": "system", "content": SYSTEM + (f"\n\nProject state:\n{context}" if context else "")}]
    for h in history[-8:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": user_message})

    # 1. native tool-calling
    try:
        import litellm
        from .llm_config import model_for
        resp = litellm.completion(model=model_for("supervisor"), messages=msgs,
                                  tools=tool_schemas(), tool_choice="auto",
                                  max_tokens=900, temperature=0.3)
        m = resp.choices[0].message
        calls = getattr(m, "tool_calls", None)
        if calls:
            c = calls[0]
            args = c.function.arguments
            if isinstance(args, str):
                args = json.loads(args or "{}")
            return {"tool": c.function.name, "args": args, "say": (m.content or "").strip()}
        return {"tool": None, "args": {}, "say": (m.content or "").strip()}
    except Exception:
        pass

    # 2. fallback: JSON planner (any provider)
    catalog = "\n".join(f"- {n}({', '.join(t['params'])}): {t['desc']}"
                        for n, t in TOOLS.items())
    try:
        from .llm import call_json
        out = call_json(
            f"{SYSTEM}\n\nProject state:\n{context}\n\nTools:\n{catalog}\n\n"
            f"User: {user_message}\n\n"
            'Reply with {"tool": name or null, "args": {...}, "say": "your reply"}.',
            role="supervisor", max_tokens=700, temperature=0.3)
        return {"tool": out.get("tool"), "args": out.get("args") or {},
                "say": out.get("say", "")}
    except Exception as e:
        return {"tool": None, "args": {}, "say": f"(couldn't reach the model: {e})"}


def needs_approval(tool: str | None) -> bool:
    return bool(tool) and TOOLS.get(tool, {}).get("writes", False)


def run_tool(tool: str, args: dict) -> Any:
    spec = TOOLS.get(tool)
    if not spec:
        return {"error": f"unknown tool {tool}"}
    try:
        return spec["fn"](**args)
    except TypeError as e:
        return {"error": f"bad arguments for {tool}: {e}"}
    except Exception as e:
        return {"error": f"{tool} failed: {e}"}


def interpret(tool: str, result: Any, user_message: str, guidance: str = "") -> str:
    """One-or-two-sentence read of a tool result, for the conversation."""
    try:
        from .llm import call_json
        out = call_json(
            f"{SYSTEM}\n\nThe user asked: {user_message}\n"
            + (f"They added this guidance: {guidance}\n" if guidance else "")
            + f"You ran `{tool}` and got:\n{json.dumps(result, default=str)[:3000]}\n\n"
            'Reply with {"say": "1-2 sentences interpreting this for the scientist, '
            'flagging any weakness in the evidence"}.',
            role="supervisor", max_tokens=400, temperature=0.3)
        return out.get("say", "")
    except Exception:
        return ""
