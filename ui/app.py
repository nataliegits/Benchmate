"""Streamlit UI for Benchmate.

Run with:
    cd ~/Desktop/Benchmate
    source .venv/bin/activate
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# Make repo imports work whether this is run from repo root or ui/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from co_scientist.tools import available_geneformer_genes, geneformer_neighbors
from co_scientist import assay, freezer
from ui.notebook_gen import generate_notebook, resolve_to_ensembl, CELL_TYPE_PRESETS
from ui.colab_handoff import handoff, gh_available, GhUnavailable
from co_scientist.llm_config import model_for, _DEFAULT_ROLE_MODELS


def _clean_reagent(raw) -> str:
    """Reduce a designed reagent line to just the compound name.

    The designer writes for a human. "Kifunensine (10 uM) - blocks mannose
    trimming", "Bortezomib, 5 nM, positive control". The freezer matcher is
    comparing against cap labels like "kifunensine", so that extra prose is
    what makes a reagent on the shelf come back as "not in this box".
    """
    import re as _re
    s = str(raw or "").strip()
    if not s:
        return ""
    # drop anything after a dash/colon separator, or a parenthetical
    # \u2014 is the em dash, \u2013 the en dash, then plain hyphen. Written as escapes
    # rather than literal characters, so this stays ASCII source while still
    # matching the dashes an LLM actually writes. (A bulk de-em-dash pass once
    # collapsed this class to [---], a malformed range that silently stopped
    # matching em/en dashes and emitted a FutureWarning.)
    s = _re.split(r"\s+[\u2014\u2013-]\s+|:\s|\s*\(", s)[0]
    # drop trailing dose / concentration fragments
    s = _re.sub(r"[,;]?\s*\d+(\.\d+)?\s*(n|u|µ|m)?[mM](ol)?\b.*$", "", s)
    s = _re.sub(r"[,;]\s*(vehicle|positive|negative)\s+control.*$", "", s,
                flags=_re.I)
    return s.strip(" ,;.").strip()


def _stale_modules() -> list[str]:
    """Detect a UI newer than the co_scientist modules backing it.

    Streamlit re-executes this file on every rerun but leaves already-imported
    modules in sys.modules. On Streamlit Cloud that means a fresh deploy can run
    a new app.py against the previous session's co_scientist package: which
    surfaces as a redacted AttributeError deep in a tab, with no hint that a
    reboot is all that's needed.

    So: name the attributes this UI depends on, and report the ones missing
    rather than letting the tab explode.
    """
    import importlib
    required = {
        "co_scientist.target_scorer": ["DEPMAP_SUMMARY", "depmap_source",
                                       "depmap_lineage_in_use"],
        "co_scientist.crosscheck": ["gene_missense_burden", "read_depmap",
                                    "model_status"],
        "co_scientist.hypothesis_scan": ["scan", "genes_in"],
    }
    missing = []
    for mod_name, attrs in required.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            missing.append(f"{mod_name} (import failed: {e})")
            continue
        gone = [a for a in attrs if not hasattr(mod, a)]
        if gone:
            missing.append(f"{mod_name}: missing {', '.join(gone)}")
    return missing


def _heal_stale_modules() -> list[str]:
    """Reload first-party modules when they're older than this UI.

    reload() mutates the existing module object in place, so anything holding a
    reference (crosscheck holds target_scorer, for instance) picks up the new
    attributes without re-importing. Order matters: dependencies first, then the
    modules that read from them.

    Safe here because these modules expose functions and constants only: no
    classes whose identity could split across a reload.
    """
    import importlib
    stale = _stale_modules()
    if not stale:
        return []
    for name in ("co_scientist.target_scorer", "co_scientist.hypothesis_scan",
                 "co_scientist.assay", "co_scientist.freezer",
                 "co_scientist.experiment", "co_scientist.crosscheck"):
        try:
            mod = sys.modules.get(name)
            if mod is not None:
                importlib.reload(mod)
        except Exception:
            pass          # a failed reload leaves the old module; guard reports it
    return _stale_modules()


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "geneformer"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Benchmate", layout="wide")

# Inter for body text, Source Serif 4 for display. Matches the editorial
# research-tool aesthetic (Elicit-style). Loaded from Google Fonts; falls
# back to system sans if blocked.
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif !important;
            font-feature-settings: 'cv11', 'ss01', 'ss03';
            letter-spacing: -0.005em;
        }
        h1, h2, h3 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif !important;
            letter-spacing: -0.02em;
            font-weight: 600;
        }
        h1 { font-size: 2.4rem !important; }
        h2 { font-size: 1.5rem !important; }
        code, pre, .stCode {
            font-family: 'JetBrains Mono', 'SF Mono', Menlo,
                         monospace !important;
        }
        /* Tab labels: bold Inter, matching the Benchmate title, at whatever
           size Streamlit already uses.

           Anchored on the ARIA role, not `.stTabs`. That class is
           emotion-generated and its name changes between Streamlit versions,
           so `.stTabs [data-baseweb="tab"]` silently matched nothing, which is
           why the first attempt at this had no visible effect. `[role="tab"]`
           is required for accessible tabs and doesn't churn. The BaseWeb
           attribute is kept as a fallback.

           Note the role selectors only ever hit tab strips, so ordinary
           buttons elsewhere in the app are unaffected.

           The descendant `*` matters too: the label text sits in a nested <p>
           inside a markdown container, and styling only the button leaves that
           <p> inheriting Streamlit's own regular weight. font-size stays
           `inherit` so this changes weight and face only. */
        [role="tab"],
        [role="tab"] *,
        [role="tablist"] button,
        [role="tablist"] button *,
        button[data-baseweb="tab"],
        button[data-baseweb="tab"] * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif !important;
            font-weight: 700 !important;
            font-size: inherit !important;
            letter-spacing: -0.01em !important;
        }
        /* selected tab in full ink, the rest a step back */
        [role="tab"][aria-selected="true"],
        [role="tab"][aria-selected="true"] * {
            color: #111111 !important;
        }
        [role="tab"][aria-selected="false"],
        [role="tab"][aria-selected="false"] * {
            color: #6b6b6b !important;
        }

        /* Monochrome editorial. Ink primary buttons */
        .stButton button[kind="primary"] {
            background-color: #111111;
            border-color: #111111;
            color: #ffffff;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #000000;
            border-color: #000000;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with a thin accent rule above the title for editorial polish
st.markdown(
    "<div style='border-top: 3px solid #111111; width: 56px; "
    "margin-bottom: 8px;'></div>",
    unsafe_allow_html=True,
)
st.title("Benchmate")
st.markdown(
    "<p style='color:#555555; font-size:1.05rem; margin-top:-8px; "
    "font-family: Inter, sans-serif;'>"
    "Your AI lab coworker. From question, to hypothesis, to bench, and back."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)


def step_intro(text: str, step: str = "", nxt: str = ""):
    """The description shown when you open a tab: what this step does to your
    hypothesis, and where it leads. Replaces a header banner. The explanation
    lives where the work is, so nothing looks clickable that isn't."""
    head = f"Step {step} · " if step else ""
    tail = (f"<div style='margin-top:8px;font-size:14px;color:#777;'>"
            f"&rarr; Next: {nxt}</div>" if nxt else "")
    st.markdown(
        f"<div style='background:#f4f4f3;border-radius:8px;padding:14px 16px;"
        f"font-size:15.5px;line-height:1.55;color:#444;margin-bottom:16px;'>"
        f"<b style='color:#111;'>{head}What this does:</b> {text}{tail}</div>",
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────
# Sidebar: keys, cache, uploads, model routing
# ────────────────────────────────────────────────────────────
from co_scientist import trace as _trace


def _run_id() -> str:
    """The run this session is recording into, created on first use."""
    if not st.session_state.get("run_id"):
        st.session_state["run_id"] = _trace.new_run(
            st.session_state.get("goal", "") or "")
    return st.session_state["run_id"]


def _tr(step: str, headline: str, **kw) -> None:
    """Record a step. Never lets a trace failure disturb the actual work."""
    try:
        _trace.record(_run_id(), step, headline, **kw)
    except Exception:
        pass


def trace_panel() -> None:
    """The running record of this session, newest last.

    Lives in the sidebar so it's visible from every tab: the point is that you
    can watch the reasoning accumulate while you work, rather than reconstruct
    it afterwards from whatever the final screen happens to show.
    """
    st.subheader("This run")
    _ids = [r["run_id"] for r in _trace.runs()]
    _cur = st.session_state.get("run_id")

    _replay = st.session_state.get("trace_replay_id")
    _show = _replay or _cur
    if not _show:
        st.caption("Nothing recorded yet. Ask a question in **Start here** and "
                   "the record builds itself as you go.")
    else:
        _evs = _trace.read(_show)
        if _replay:
            # Demo mode: step through a finished run at your own pace, so a
            # recorded walkthrough has repeatable timing and no live latency.
            _n = st.slider("Replay step", 1, max(len(_evs), 1),
                           min(len(_evs), st.session_state.get("trace_step", 1)),
                           key="trace_step")
            _evs = _evs[:_n]
            st.caption(f"Replaying `{_replay}`, step {_n} of "
                       f"{len(_trace.read(_replay))}.")
        if not _evs:
            st.caption("Recording. Nothing has happened yet.")
        for _e in _evs:
            st.markdown(
                f"**{_trace.STEP_LABEL.get(_e['step'], _e['step'])}**  \n"
                f"{_e['headline']}")
            if _e.get("detail"):
                st.caption(_e["detail"][:280])
            for _k, _v in (_e.get("outputs") or {}).items():
                st.caption(f"{_k}: {_v}")
            st.divider()

        _f = _trace.folder(_show)
        _files = sorted(p.name for p in (_f / "files").glob("*")) \
            if (_f / "files").exists() else []
        with st.expander(f"Run folder ({len(_files)} file(s))"):
            st.caption(f"`{_f}`")
            for _n2 in _files:
                st.caption(_n2)
            st.download_button("Download the run summary",
                               _trace.summarise(_show),
                               file_name=f"benchmate_{_show}.txt",
                               key=f"tr_dl_{_show}", use_container_width=True)

    with st.expander("Past runs"):
        if not _ids:
            st.caption("No past runs yet.")
        for _r in _trace.runs()[:8]:
            st.caption(f"`{_r['run_id']}` · {_r['n_events']} steps · "
                       f"{(_r.get('question') or '(no question)')[:48]}")
        _pick = st.selectbox("Replay a run", ["(off)"] + _ids, key="trace_pick")
        st.session_state["trace_replay_id"] = (
            None if _pick == "(off)" else _pick)
        if st.button("Start a fresh run", key="tr_new",
                     use_container_width=True):
            st.session_state.pop("run_id", None)
            st.session_state.pop("trace_replay_id", None)
            st.rerun()


with st.sidebar:
    trace_panel()
    st.divider()
    st.subheader("Anthropic API key")
    # CRITICAL: do NOT pre-populate this field from os.environ. Even with
    # type="password" the rendered DOM value is visible to anyone using
    # browser DevTools. Each user enters their own key from scratch; the
    # value lives only in their session.
    if "anthropic_key" not in st.session_state:
        st.session_state.anthropic_key = ""
    user_key = st.text_input(
        "Your Anthropic key",
        type="password",
        placeholder="sk-ant-...",
        key="anthropic_key",
        help="Available at console.anthropic.com. Stored only in your "
             "browser session; never sent to the Benchmate maintainer.",
    )
    if user_key:
        # Only set the env var for this Python process if the user provided
        # a key in this session. Never fall back to a maintainer-set env
        # var that would leak across sessions.
        os.environ["ANTHROPIC_API_KEY"] = user_key
        st.caption("Key set for this session.")
    else:
        # Make sure no leaked key from a previous session lingers
        os.environ.pop("ANTHROPIC_API_KEY", None)
        st.caption("Required for the Run Benchmate tab.")

    st.divider()
    st.subheader("Cached Geneformer perturbations")
    cached = available_geneformer_genes()
    if cached:
        st.metric("genes", len(cached))
        with st.container(border=True):
            for g in cached:
                st.markdown(f"`{g}`")
    else:
        st.info("No cached perturbations yet.")

    st.divider()
    st.subheader("Upload CSVs")
    uploaded = st.file_uploader(
        "Drop *_stats.csv files here",
        type=["csv"],
        accept_multiple_files=True,
        help="After running the Colab notebook, drag the downloaded "
             "*_stats.csv files into this box.",
    )
    if uploaded:
        for f in uploaded:
            name = f.name
            if not name.endswith("_stats.csv"):
                st.warning(f"Skipping {name} (expected *_stats.csv)")
                continue
            dst = CACHE_DIR / name
            dst.write_bytes(f.read())
            st.success(f"Cached {name}")
        _tr("cache", f"Loaded {len(uploaded)} perturbation file(s)",
            outputs={"genes": ", ".join(
                f.name.replace("_stats.csv", "") for f in uploaded)},
            files=[str(CACHE_DIR / f.name) for f in uploaded])
        st.rerun()

    # Colab writes the CSVs to your Downloads folder, and dragging them back in
    # is the most annoying step in the loop. Look for them instead. Only helps
    # when Benchmate runs on the same machine as the browser, so it's offered
    # rather than assumed.
    _dl = Path.home() / "Downloads"
    if _dl.exists():
        _found = sorted(_dl.glob("*_stats.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:12]
        _new = [p for p in _found if not (CACHE_DIR / p.name).exists()]
        if _new:
            st.caption(f"Found {len(_new)} new result file(s) in your Downloads "
                       f"folder, straight from Colab.")
            if st.button(f"Import {len(_new)} file(s)", key="colab_import",
                         use_container_width=True):
                _ok = []
                for _p in _new:
                    try:
                        shutil.copy2(_p, CACHE_DIR / _p.name)
                        _ok.append(_p.name)
                    except Exception as _e:
                        st.warning(f"Couldn't import {_p.name}: {_e}")
                if _ok:
                    st.success(f"Imported {', '.join(_ok)}")
                    _tr("cache", f"Imported {len(_ok)} file(s) back from Colab",
                        outputs={"genes": ", ".join(
                            n.replace("_stats.csv", "") for n in _ok)},
                        files=[str(CACHE_DIR / n) for n in _ok])
                    st.rerun()

    st.divider()
    with st.expander("Model routing", expanded=False):
        st.caption(
            "Assign a model to each agent role. The default sends "
            "reasoning-heavy roles to Sonnet and throughput roles to Haiku. "
            "Changes apply to the next Run Benchmate call."
        )
        MODEL_OPTIONS = [
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-opus-4-7",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash",
        ]
        ROLE_PRICE = {
            "anthropic/claude-sonnet-4-6": 3.0,
            "anthropic/claude-haiku-4-5": 1.0,
            "anthropic/claude-opus-4-7": 15.0,
            "openai/gpt-5": 2.5,
            "openai/gpt-5-mini": 0.25,
            "gemini/gemini-2.5-pro": 1.25,
            "gemini/gemini-2.5-flash": 0.30,
        }
        ROLES = ["generation", "reflection", "evolution",
                 "ranking", "meta_review", "supervisor"]
        for role in ROLES:
            current = model_for(role)
            if current not in MODEL_OPTIONS:
                MODEL_OPTIONS.insert(0, current)
            idx = MODEL_OPTIONS.index(current)
            choice = st.selectbox(
                role,
                MODEL_OPTIONS,
                index=idx,
                key=f"model_{role}",
                help=f"~${ROLE_PRICE.get(current, '?')}/1M input tokens",
            )
            if choice != current:
                os.environ[f"BENCHMATE_MODEL_{role.upper()}"] = choice
        st.caption(
            "Generation, Reflection, and Evolution benefit most from a "
            "strong model, because they read the Geneformer evidence. The others "
            "can run cheaper without measurable quality loss."
        )

# ────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────
DEFAULT_GOAL = (
    "Generate testable hypotheses for how TXNDC15 couples ERAD to "
    "mitophagy in human cells. Use SYVN1 and MARCHF6 as known-comparator "
    "ERAD E3 ligases. Prioritise experiments feasible in HEK293 or "
    "HepG2 with standard proteostasis tools (CCCP, thapsigargin, "
    "tunicamycin, cycloheximide, co-IP, immunoblotting)."
)
_PRESET_KEYS = list(CELL_TYPE_PRESETS.keys())
# Defaults for the inputs the "Start here" tab pre-fills in the other tabs.
st.session_state.setdefault("genes_in", "")
st.session_state.setdefault("preset_name", _PRESET_KEYS[0])
st.session_state.setdefault("goal", DEFAULT_GOAL)
st.session_state.setdefault("run_iterations", 8)
st.session_state.setdefault("sh_plan", None)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · Start here",
    "2 · Add evidence",
    "3 · Generate & rank",
    "4 · Stress-test",
    "5 · Run experiment",
    "About",
])


# ── Benchmate anchor. One agent, reachable from every tab ───
def benchmate_anchor(where: str, suggestions: str = ""):
    """A collapsible Benchmate assistant pinned into a tab.

    One shared conversation and one shared approval queue across every tab, so
    it's the same colleague wherever you are. It just also knows which page
    you're on. Reads run instantly; anything that spends credits or writes
    evidence waits for you.
    """
    from co_scientist import orchestrator as orch

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("pending", None)
    pend = st.session_state.pending
    n = len(st.session_state.chat)
    label = ("Benchmate. Waiting for your approval" if pend
             else f"Ask Benchmate{f'  ({n})' if n else ''}")

    with st.expander(label, expanded=bool(pend)):
        st.caption(
            "This is the assistant, not a form. It can see your whole project: "
            "the leaderboard, bench results, and freezer, so use it to *ask about* "
            "or *steer* this step, in plain language. (The fields above are the "
            "actual inputs.)"
        )
        if suggestions:
            st.caption(f"Try: {suggestions}")

        for m in st.session_state.chat[-6:]:
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.markdown(m["content"])
                if m.get("data") is not None:
                    with st.expander("result"):
                        st.json(m["data"], expanded=False)

        # ---- approval gate (the human-in-the-loop moment) ----
        if pend:
            st.warning(f"**Waiting for you.** I'd like to run `{pend['tool']}`.")
            if pend.get("say"):
                st.markdown(pend["say"])
            st.json(pend["args"], expanded=False)
            note = st.text_input("Add guidance before it runs (optional)",
                                 key=f"note_{where}",
                                 placeholder="e.g. use a 6-point dose series")
            c1, c2, c3 = st.columns(3)
            if c1.button("Approve & run", type="primary", key=f"go_{where}"):
                args = dict(pend["args"])
                if note.strip():
                    for k in ("hypothesis", "result_summary"):
                        if k in args:
                            args[k] = f"{args[k]}\n\nUser guidance: {note.strip()}"
                            break
                with st.spinner(f"Running {pend['tool']}…"):
                    res = orch.run_tool(pend["tool"], args)
                    say = orch.interpret(pend["tool"], res, pend["user_msg"], note)
                st.session_state.chat.append(
                    {"role": "assistant",
                     "content": say or f"Ran `{pend['tool']}`.", "data": res})
                if pend["tool"] == "design_experiment" and isinstance(res, dict) \
                        and not res.get("error"):
                    st.session_state.loop_design = res
                    st.session_state.chat.append(
                        {"role": "assistant",
                         "content": "I put the design in **Experiment → Design**."})
                if pend["tool"] == "refine_hypothesis" and isinstance(res, dict):
                    st.session_state.loop_refined = res
                st.session_state.pending = None
                st.rerun()
            if c2.button("Skip", key=f"skip_{where}"):
                st.session_state.pending = None
                st.rerun()
            if c3.button("Clear chat", key=f"clr_{where}"):
                st.session_state.chat = []
                st.session_state.pending = None
                st.rerun()

        # ---- ask ----
        q = st.text_input("Ask, steer, or correct Benchmate…", key=f"ask_{where}")
        if st.button("Send", key=f"send_{where}") and q.strip():
            st.session_state.chat.append({"role": "user", "content": q.strip()})
            if not os.environ.get("ANTHROPIC_API_KEY"):
                st.session_state.chat.append(
                    {"role": "assistant",
                     "content": "Add your Anthropic API key in the sidebar and I'll get going."})
                st.rerun()
            ctx = _project_context() + f"\nThe user is currently on the **{where}** tab."
            with st.spinner("Thinking…"):
                plan = orch.decide(q.strip(), st.session_state.chat[:-1], ctx)
            tool = plan.get("tool")
            if tool and orch.needs_approval(tool):
                st.session_state.pending = {"tool": tool, "args": plan.get("args", {}),
                                            "say": plan.get("say", ""),
                                            "user_msg": q.strip()}
            elif tool:
                res = orch.run_tool(tool, plan.get("args", {}))
                say = (orch.interpret(tool, res, q.strip())
                       or plan.get("say") or f"Ran `{tool}`.")
                st.session_state.chat.append(
                    {"role": "assistant", "content": say, "data": res})
            else:
                st.session_state.chat.append(
                    {"role": "assistant", "content": plan.get("say") or "…"})
            st.rerun()


def _project_context() -> str:
    """What Benchmate knows about the project, injected on every turn."""
    from co_scientist import orchestrator as orch
    bits = []
    lb = orch.run_tool("show_leaderboard", {"top_n": 3})
    if lb.get("hypotheses"):
        bits.append("Top hypotheses: " + "; ".join(
            f"{h['statement'][:80]} (Elo {h['elo']})" for h in lb["hypotheses"]))
    br = orch.run_tool("bench_results", {})
    if br.get("results"):
        bits.append("Bench results: " + "; ".join(
            f"{r['label']} → {r['verdict']} ({r['action']})" for r in br["results"]))
    bits.append("Freezer inventory loaded: "
                + st.session_state.get("loop_box_name", "demo_drug_box.csv"))
    return "\n".join(bits)

# ── Tab 0. Guided flow ──────────────────────────────────────
with tab0:
    step_intro("takes your research question and lays out which genes to perturb, in which cells, and which models to check against. Then pre-fills the other tabs.", "1 of 5", "Add evidence")
    st.header("Start here: from question to cross-checked hypotheses")
    st.write(
        "Type a research question. Benchmate reads it and lays out the exact "
        "steps: which genes to perturb, in which cell type, and which models "
        "to cross-check the winning hypotheses against."
    )

    st.session_state.setdefault(
        "sh_question",
        "What ERAD genes drive bortezomib resistance in multiple myeloma?",
    )
    question = st.text_area("Your research question", key="sh_question", height=90)

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not have_key:
        st.info("Add your Anthropic API key in the sidebar to build a plan.")

    if st.button("Build my plan", type="primary", disabled=not have_key):
        from co_scientist.llm import call_json
        presets = list(CELL_TYPE_PRESETS.keys())
        prompt = (
            "You are planning a Benchmate run for a biomedical research "
            "question. Decide:\n"
            "1. perturb_genes: 2-4 human gene SYMBOLS most worth perturbing "
            "to generate evidence for this question.\n"
            f"2. cell_context: choose EXACTLY one string from this list: "
            f"{presets}.\n"
            "3. cell_reason: <=15 words on why that cell context fits.\n"
            "4. iterations: an integer 4-16 for the tournament size.\n"
            "5. cross_check: 2-4 items, each an object with keys target (a "
            "gene or protein symbol the hypotheses will likely center on), "
            "model (EXACTLY one of: AlphaGenome, Boltz, Open Targets, DepMap, "
            "AlphaMissense), and reason (<=10 words). Match the model to the "
            "target: Open Targets/DepMap for gene-disease questions, Boltz for "
            "drug-target binding, AlphaGenome for regulatory variants, "
            "AlphaMissense for coding variants.\n\n"
            f"Question: {question}"
        )
        try:
            with st.spinner("Reading your question…"):
                plan = call_json(prompt, role="generation", max_tokens=900)
            if plan.get("cell_context") not in CELL_TYPE_PRESETS:
                plan["cell_context"] = presets[0]
            st.session_state.sh_plan = plan
        except Exception as e:
            st.error(f"Could not build a plan: {e}")

    plan = st.session_state.get("sh_plan")
    if plan:
        genes = [str(g).upper() for g in plan.get("perturb_genes", []) if str(g).strip()]
        cell = plan.get("cell_context", _PRESET_KEYS[0])
        iters = max(4, min(16, int(plan.get("iterations", 8) or 8)))
        xchecks = plan.get("cross_check", []) or []

        st.divider()
        st.caption("Your path: three steps. Each fills the right tab for you.")

        with st.container(border=True):
            st.markdown("**1 · Generate your evidence**: New perturbation tab")
            st.markdown(
                f"Perturb **{', '.join(genes) or '-'}** in **{cell}**, "
                "then download and run the notebook."
            )
            if plan.get("cell_reason"):
                st.caption(f"Why this cell context: {plan['cell_reason']}")
            if st.button("Prefill the New perturbation tab", key="sh_fill1"):
                st.session_state.genes_in = ", ".join(genes)
                st.session_state.preset_name = cell
                st.success("Filled. Open the New perturbation tab above.")

        with st.container(border=True):
            st.markdown("**2 · Generate & rank hypotheses**: Run Benchmate tab")
            st.markdown(
                f"Run **{iters} iterations** on your question. The agents "
                "propose hypotheses and rank them by Elo."
            )
            if st.button("Prefill the Run Benchmate tab", key="sh_fill2"):
                # Name the planned genes in the goal. The agents match cached
                # perturbation data by gene symbol, so a goal phrased as
                # "what ERAD-pathway genes..." would otherwise never surface
                # the very data this plan just picked.
                goal_text = question
                if genes:
                    goal_text += ("\n\nFocus genes (perturbation data available): "
                                  + ", ".join(genes) + ".")
                st.session_state.goal = goal_text
                st.session_state.run_iterations = iters
                st.success("Filled: including your focus genes, so the run "
                           "picks up their perturbation data.")

        with st.container(border=True):
            st.markdown("**3 · Cross-check the winners**: Cross-check tab")
            if xchecks:
                for x in xchecks:
                    st.markdown(
                        f"- **{str(x.get('target', '?')).upper()}** → "
                        f"*{x.get('model', '?')}*  ({x.get('reason', '')})"
                    )
            else:
                st.markdown("Score your top genes/proteins against the panel.")
            st.caption("Low agreement with the Elo ranking = a flag before the bench.")
            st.session_state.sh_crosscheck = xchecks

        if st.button("Prefill steps 1 & 2", key="sh_fill_all"):
            st.session_state.genes_in = ", ".join(genes)
            st.session_state.preset_name = cell
            goal_text = question
            if genes:
                goal_text += ("\n\nFocus genes (perturbation data available): "
                              + ", ".join(genes) + ".")
            st.session_state.goal = goal_text
            st.session_state.run_iterations = iters
            st.success("Both tabs filled. Work through them top to bottom.")

    st.divider()
    benchmate_anchor("start", "*focus on less-studied ERAD genes* · *why did you pick these genes?* · *suggest a sharper version of my question*")

# ── Tab 1 ────────────────────────────────────────────────────
with tab1:
    step_intro("adds your own experimental data (Geneformer perturbations) so the agents reason from your bench, not just the literature.", "2 of 5", "Generate &amp; rank")
    st.header("Add genes to the perturbation cache")
    st.write(
        "Enter the gene symbols you want to perturb and pick the cell "
        "context. Benchmate generates a Colab notebook with both pre-filled."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        genes_in = st.text_input(
            "Gene symbols (comma-separated)",
            placeholder="e.g. XBP1, ATF4, EIF2AK3",
            key="genes_in",
        )
    with col2:
        preset_name = st.selectbox(
            "Cell context",
            _PRESET_KEYS,
            key="preset_name",
            help="The cells Geneformer will perturb your genes in. "
                 "Pick where your genes' biology should be readable.",
        )
    if st.session_state.get("sh_plan"):
        st.caption("Tip: the Start here tab can fill these in from a question.")
    st.caption(CELL_TYPE_PRESETS[preset_name]["rationale"])

    if st.button("Generate Colab notebook", type="primary"):
        if not genes_in.strip():
            st.error("Enter at least one gene symbol.")
        else:
            symbols = [g.strip().upper() for g in genes_in.split(",") if g.strip()]
            with st.spinner("Resolving Ensembl IDs..."):
                try:
                    resolved = resolve_to_ensembl(symbols)
                except Exception as e:
                    st.error(f"mygene lookup failed: {e}")
                    st.stop()
            unresolved = [s for s in symbols if s not in resolved]
            if unresolved:
                st.warning(f"Could not resolve: {', '.join(unresolved)}")
            if not resolved:
                st.error("No symbols resolved. Check spelling.")
                st.stop()

            st.success(f"Resolved {len(resolved)} gene(s) for context: {preset_name}")
            st.json(resolved)

            with st.spinner("Generating notebook..."):
                nb_path, _ = generate_notebook(resolved.keys(), preset_name=preset_name)
            st.write(f"Notebook saved: `{nb_path.relative_to(REPO_ROOT)}`")

            if gh_available():
                with st.spinner("Pushing to Gist..."):
                    try:
                        urls = handoff(nb_path, description=f"Benchmate: {', '.join(resolved)}")
                    except Exception as e:
                        urls = None
                        st.warning(f"Gist push failed: {e}. Falling back to download.")
                if urls:
                    st.markdown(f"### [Open in Colab]({urls['colab_url']})")
                    st.caption(f"Gist: {urls['gist_url']}")
            else:
                st.info(
                    "GitHub CLI not available. Download the notebook below, "
                    "open colab.research.google.com, and use "
                    "File → Upload notebook to load it."
                )
                st.download_button(
                    "Download notebook",
                    nb_path.read_bytes(),
                    file_name=nb_path.name,
                    mime="application/x-ipynb+json",
                )

            st.divider()
            st.info(
                "In Colab: Runtime → Change runtime type → T4 GPU, then "
                "Runtime → Run all. When the notebook finishes, the final "
                "cell auto-downloads each *_stats.csv to your browser. "
                "Drag those files into the Upload CSVs panel in the sidebar."
            )

    st.divider()
    benchmate_anchor("evidence", "*what perturbation data do I have?* · *what else should I gather?*")

# ── Tab 2. Run Benchmate ────────────────────────────────────
with tab2:
    step_intro("seven agents propose hypotheses and argue; an Elo tournament ranks them, so you get a shortlist instead of a wall of text.", "3 of 5", "Stress-test")
    st.header("Run the Co-Scientist loop")
    goal = st.text_area(
        "Research goal",
        height=140,
        key="goal",
    )
    iterations = st.slider("Max iterations", 4, 16, key="run_iterations")
    cost_estimate = 0.30 * iterations / 4
    st.caption(f"Estimated Anthropic spend: ~${cost_estimate:.2f} "
               f"({iterations} iterations).")

    # A run takes minutes. Two things made that look like a dead button:
    # the child's stdout is block-buffered when piped, so nothing appeared for
    # ages, and reading it line-by-line blocked the whole Streamlit script. # freezing the page. So: launch detached, log to a file, and poll.
    RUN_LOG = REPO_ROOT / ".benchmate_run.log"
    _proc = st.session_state.get("run_proc")
    _running = _proc is not None and _proc.poll() is None

    c_run, c_stop = st.columns([1, 1])
    _go = c_run.button("Run Benchmate", type="primary", disabled=_running)
    if c_stop.button("Stop run", disabled=not _running):
        _proc.terminate()
        st.session_state.run_proc = None
        st.warning("Run stopped.")

    if _go:
        if not goal.strip():
            st.warning("Type a research goal first.")
        elif not (os.environ.get("ANTHROPIC_API_KEY")
                  or (REPO_ROOT / ".env").exists()):
            st.error("No ANTHROPIC_API_KEY found. Add it in the sidebar or "
                     "your .env file. The run would fail immediately.")
        else:
            _fh = open(RUN_LOG, "w")
            st.session_state.run_proc = subprocess.Popen(
                # -u and PYTHONUNBUFFERED so the log appears as it happens
                # instead of arriving in one lump when the process exits
                [sys.executable, "-u", "run.py", goal,
                 "--max-iterations", str(iterations)],
                cwd=REPO_ROOT, stdout=_fh, stderr=subprocess.STDOUT,
                text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            st.session_state.run_started = time.time()
            st.session_state.pop("run_ok", None)
            st.rerun()

    # ---- poll the running process (survives tab switches) -------------------
    _proc = st.session_state.get("run_proc")
    if _proc is not None:
        _tail = RUN_LOG.read_text()[-6000:] if RUN_LOG.exists() else ""
        _rc = _proc.poll()
        if _rc is None:
            _el = int(time.time() - st.session_state.get("run_started", time.time()))
            st.info(f"Running: {_el // 60}m {_el % 60}s elapsed. "
                    f"This refreshes itself; you can switch tabs and come back.")
            st.code(_tail or "starting up (loading models, first API call)…")
            time.sleep(2.5)
            st.rerun()
        else:
            st.session_state.run_ok = (_rc == 0)
            st.session_state.run_rc = _rc
            st.session_state.run_proc = None
            if _rc == 0:
                try:
                    _sd = json.loads((REPO_ROOT / "state.json").read_text())
                    _hs = sorted(_sd.get("hypotheses", []),
                                 key=lambda h: h.get("elo", 0), reverse=True)
                    _tr("generate", f"{len(_hs)} hypotheses proposed",
                        detail=goal,
                        outputs={"top": (_hs[0].get("statement", "")[:150]
                                         if _hs else "-")},
                        files=[str(REPO_ROOT / "state.json")])
                    _tr("rank", "Elo tournament settled",
                        outputs={"leader": f"{round(_hs[0].get('elo', 0))}"
                                           if _hs else "-",
                                 "matches": sum(h.get("matches_played", 0)
                                                for h in _hs)})
                except Exception:
                    pass
            with st.expander("Run log", expanded=_rc != 0):
                st.code(_tail or "(no output)")
            st.rerun()

    # ---- Results (rendered on EVERY rerun, so they survive clicking around) ----
    # Read from state.json on disk, so the leaderboard is still here after you
    # switch tabs, tweak a widget, or even restart the app.
    state_file = REPO_ROOT / "state.json"
    if st.session_state.get("run_ok") is False:
        st.error(f"Benchmate exited with code {st.session_state.get('run_rc')}. "
                 f"The log below says why.")
    elif st.session_state.get("run_ok"):
        st.success("Benchmate finished.")
    # keep the log reachable after the run ends. It's the only place a crash
    # or a rate-limit message shows up
    if RUN_LOG.exists() and st.session_state.get("run_proc") is None:
        with st.expander("Run log", expanded=st.session_state.get("run_ok") is False):
            st.code(RUN_LOG.read_text()[-8000:] or "(empty)")
    if state_file.exists():
        # Parse and display top hypotheses as a real Streamlit table
        # so column headers stay visible regardless of log scrollback.
        state_data = json.loads(state_file.read_text())
        hyps = sorted(
            state_data.get("hypotheses", []),
            key=lambda h: (h.get("matches_played", 0) > 0,
                           h.get("elo", 0)),
            reverse=True,
        )[:5]
        if hyps:
            import pandas as pd
            rows = []
            for h in hyps:
                played = h.get("matches_played", 0)
                rows.append({
                    "Elo": round(h.get("elo", 0), 0),
                    "Matches": played,
                    "Gen": h.get("generation", 0),
                    "Ranked": "yes" if played > 0 else "no (untested)",
                    "Statement": h.get("statement", ""),
                })
            df = pd.DataFrame(rows)
            st.subheader(f"Top {len(hyps)} hypotheses")
            st.caption("Loaded from the last completed run "
                       "(`state.json`): this persists across tab "
                       "switches and app restarts.")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Elo": st.column_config.NumberColumn(
                        width="small",
                        help="Tournament rating (starts at 1200). "
                             "Wins push up, losses pull down."),
                    "Matches": st.column_config.NumberColumn(
                        width="small",
                        help="Number of pairwise tournament rounds "
                             "this hypothesis has been judged in. "
                             "Higher = more reliable rating."),
                    "Gen": st.column_config.NumberColumn(
                        width="small",
                        help="Generation: 0 = original Generation "
                             "output, 1 = first Evolution refinement, "
                             "2 = second, etc."),
                    "Ranked": st.column_config.TextColumn(
                        width="small",
                        help="'no' means the hypothesis was created "
                             "by Evolution after the last Ranking "
                             "round, so its Elo is still the default "
                             "1200 and isn't a quality signal."),
                    "Statement": st.column_config.TextColumn(width="large"),
                },
            )
            # Per-hypothesis expand for rationale + experiment
            for i, h in enumerate(hyps, 1):
                with st.expander(
                    f"#{i}: Elo {round(h.get('elo', 0), 0):.0f} "
                    f"rationale + experiment"
                ):
                    st.markdown("**Rationale**")
                    st.markdown(h.get("rationale", ""))
                    st.markdown("**Proposed experiment**")
                    st.markdown(h.get("experiment", ""))
                    if h.get("review_notes"):
                        st.markdown("**Reviewer notes**")
                        for note in h["review_notes"]:
                            st.markdown(f"- {note}")
        st.download_button(
            "Download state.json",
            state_file.read_text(),
            file_name="state.json",
        )

    st.divider()
    benchmate_anchor("leaderboard", "*summarise the top 3 for me* · *why did CB-5083 drop?* · *which of these is most novel?*")

# ── Benchmark section (folded into "Cross-check your hypotheses") ────
def _benchmark_section():
    st.header("Is the Elo leaderboard trustworthy?")
    st.write(
        "The simulator replays Benchmate's **real** `elo.py` against synthetic "
        "hypotheses whose true quality we control. For free, thousands of "
        "times, so you can see how match budget, K-factor, and judge skill "
        "move ranking accuracy and repeatability before spending API credits."
    )
    st.caption(
        "Reads as: **spearman**→ranking accuracy vs ground truth "
        "(1.0=perfect), **top1**→true best ends #1, **repeat**→two runs pick "
        "the same winner, **churn**→rank movement in the final cycle "
        "(→0 = converged)."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sim_n = st.slider("Hypotheses (n)", 4, 12, 6,
                          help="How many hypotheses are in the simulated tournament.")
    with col_b:
        sim_skill = st.slider("Judge skill", 0.55, 0.90, 0.70, 0.05,
                              help=("P(judge picks the better of two adjacent "
                                    "hypotheses). 0.5 = coin flip; ~0.70 is "
                                    "decent. Measure your real judge with "
                                    "`judge-eval` and feed the number back in."))
    with col_c:
        sim_replicates = st.select_slider(
            "Replicates", options=[50, 100, 200, 300], value=200,
            help="More replicates = tighter error bars but a slower run."
        )

    st.markdown("**Pick a study**: what question do you want the simulator to answer?")
    study = st.radio(
        "Study",
        ["Match budget: how many matches do I need?",
         "K-factor: does the 40/20/10 schedule matter?",
         "Judge quality: how badly does a weak / biased judge hurt?"],
        label_visibility="collapsed",
    )

    if st.button("Run simulator", type="primary"):
        from benchmark.simulate import (evaluate, CURRENT_K, k_const,
                                         k_schedule)

        rows: list[dict] = []
        with st.spinner("Replaying tournaments…"):
            if study.startswith("Match budget"):
                configs = [(1, 6, "1 round (12 matches)"),
                           (2, 6, "2 rounds (24 matches)"),
                           (4, 6, "4 rounds (48 matches)"),
                           (6, 8, "6 rounds, 8/cycle (96)"),
                           (10, 10, "10 rounds, 10/cycle (200)")]
                for cycles, npc, lbl in configs:
                    s = evaluate(lbl, replicates=sim_replicates,
                                 n=sim_n, cycles=cycles, n_per_cycle=npc,
                                 judge_skill=sim_skill)
                    rows.append({
                        "config": s.label,
                        "matches/hyp": round(s.matches_per_hyp, 1),
                        "spearman": round(s.mean_spearman, 2),
                        "± sd": round(s.sd_spearman, 2),
                        "top1": f"{s.top1_accuracy:.0%}",
                        "top3": f"{s.top3_accuracy:.0%}",
                        "repeat": f"{s.repeatability:.0%}",
                        "churn": round(s.final_churn, 2),
                    })
            elif study.startswith("K-factor"):
                configs = [("current 40/20/10", CURRENT_K),
                           ("constant K=32 (FIDE)", k_const(32)),
                           ("constant K=16", k_const(16)),
                           ("gentle 24/16/8",
                            k_schedule([(5, 24), (15, 16)], 8)),
                           ("hot 64/32/16",
                            k_schedule([(5, 64), (15, 32)], 16))]
                for lbl, k_fn in configs:
                    s = evaluate(lbl, replicates=sim_replicates,
                                 n=sim_n, cycles=4, n_per_cycle=6,
                                 judge_skill=sim_skill, k_fn=k_fn)
                    rows.append({
                        "config": s.label,
                        "matches/hyp": round(s.matches_per_hyp, 1),
                        "spearman": round(s.mean_spearman, 2),
                        "± sd": round(s.sd_spearman, 2),
                        "top1": f"{s.top1_accuracy:.0%}",
                        "top3": f"{s.top3_accuracy:.0%}",
                        "repeat": f"{s.repeatability:.0%}",
                        "churn": round(s.final_churn, 2),
                    })
            else:  # judge quality
                configs = [("strong (skill 80%)", 0.80, 0.0),
                           ("decent (skill 70%)", 0.70, 0.0),
                           ("weak (skill 60%)", 0.60, 0.0),
                           ("decent + 15% position bias", 0.70, 0.15)]
                for lbl, skill_v, pbias in configs:
                    s = evaluate(lbl, replicates=sim_replicates,
                                 n=sim_n, cycles=4, n_per_cycle=6,
                                 judge_skill=skill_v, position_bias=pbias)
                    rows.append({
                        "config": s.label,
                        "matches/hyp": round(s.matches_per_hyp, 1),
                        "spearman": round(s.mean_spearman, 2),
                        "± sd": round(s.sd_spearman, 2),
                        "top1": f"{s.top1_accuracy:.0%}",
                        "top3": f"{s.top3_accuracy:.0%}",
                        "repeat": f"{s.repeatability:.0%}",
                        "churn": round(s.final_churn, 2),
                    })

        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        if study.startswith("Match budget"):
            st.info(
                "**Read:** top-3 is reliable around ~8 matches/hyp; a "
                "trustworthy *#1* needs ~12+. Your `state.json` today shows "
                "~2 matches/hyp."
            )
        elif study.startswith("K-factor"):
            st.info(
                "**Read:** at this scale the K-factor barely moves the "
                "metrics. The 40/20/10 schedule, flat K=16, and flat K=32 "
                "are within noise. Match budget is the lever, not K."
            )
        else:
            st.info(
                "**Read:** judge skill is the accuracy ceiling. Improving "
                "the judge buys as much as doubling the match budget. The "
                "fair judge (default) neutralises position bias by judging "
                "each pair in both orders."
            )

    st.divider()
    st.markdown("### Live benchmarks (real API calls)")

    from benchmark import results as bench_results
    IS_HOSTED = bool(os.environ.get("BENCHMATE_HOSTED"))

    def _show_saved(name: str) -> None:
        """Render the latest saved run for a benchmark (params + metric tiles).
        On the hosted demo this is the only thing shown for a section."""
        run = bench_results.latest(name)
        if not run:
            if IS_HOSTED:
                st.caption("No saved run yet. Run this locally to populate the "
                           "hosted demo.")
            return
        p = run.get("params", {})
        bits = ", ".join(f"{k}={v}" for k, v in p.items())
        st.caption(f"📊 Saved local run{(' · ' + bits) if bits else ''} · "
                   f"captured {run.get('captured', '?')}")
        metrics = run.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k, v)
        if run.get("note"):
            st.caption(run["note"])

    if IS_HOSTED:
        st.info(
            "**Hosted demo.** Each section shows results captured on a local "
            "run (saved in the repo). Want to run them live yourself? Enter "
            "**your own** Anthropic key in the sidebar. You pay only for your "
            "own calls, and the key-only benchmarks below unlock. The ontology "
            "comparison additionally needs a local OntoMCP server (see its "
            "section), so it can't run on the hosted site."
        )
    else:
        st.caption(
            "These run on the ERAD gold set in `benchmark/gold_set.py` and need "
            "an Anthropic key (set in the sidebar). Ranking calls use Haiku, "
            "so the spend is small. The estimates below are upper bounds. "
            "Each run is saved to `benchmark/results/`, which is what the "
            "hosted demo displays."
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning(
            "Add an Anthropic key in the sidebar to run any of these live. "
            "Without one, the saved results above each section are still shown."
        )

    # ---- 1. Judge accuracy ------------------------------------------------
    with st.container(border=True):
        st.markdown(
            "**1. Judge accuracy**: is the LLM judge any good? "
            "Measures accuracy on clear cross-tier pairs, position-bias rate "
            "(how often the verdict flips when you swap A/B), self-consistency, "
            "and transitivity violations."
        )
        _show_saved("judge_eval")
        je_pairs = st.slider("Cross-tier pairs to test", 4, 12, 8,
                             key="je_pairs",
                             help="More pairs = tighter estimates but more spend.")
        je_cost = 0.001 * (je_pairs * 4 + 30)        # ~4 calls/pair + transitivity sweep
        st.caption(f"Estimated Anthropic spend: ~${je_cost:.2f}. "
                   f"Expected runtime: ~{max(1, je_pairs // 2)}-"
                   f"{je_pairs} minutes.")
        if st.button("Run judge-eval", type="primary",
                     disabled=not os.environ.get("ANTHROPIC_API_KEY"),
                     key="je_btn"):
            from benchmark.judge_eval import evaluate_judge
            with st.spinner("Judging gold-set pairs in both orders…"):
                r = evaluate_judge(max_pairs=int(je_pairs), role="ranking")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("accuracy", f"{r.accuracy:.0%}",
                      help="Want > 80% on clear pairs.")
            c2.metric("position bias", f"{r.position_bias_rate:.0%}",
                      help="Want < 15%. Verdicts that flip with A/B order.")
            c3.metric("self-consistency", f"{r.self_consistency:.0%}",
                      help="Want > 85%. Same call twice agrees.")
            tv = (f"{r.transitivity_violations}/{r.transitivity_triples}"
                  if r.transitivity_triples else "n/a")
            c4.metric("transitivity", tv,
                      help="Want ~0 cycles (A>B>C>A).")
            bench_results.save_run("judge_eval", {
                "label": "Judge accuracy",
                "params": {"pairs": int(je_pairs)},
                "metrics": {"accuracy": f"{r.accuracy:.0%}",
                            "position bias": f"{r.position_bias_rate:.0%}",
                            "self-consistency": f"{r.self_consistency:.0%}",
                            "transitivity": tv},
            })
            if r.accuracy < 0.8 or r.position_bias_rate > 0.15:
                st.info(
                    "The judge is the bottleneck. The fair judge (default) "
                    "is already neutralising bias; a stronger ranking model "
                    "(Sonnet via `BENCHMATE_MODEL_RANKING`) is the next "
                    "lever."
                )

    # ---- 2. Validate vs gold ---------------------------------------------
    with st.container(border=True):
        st.markdown(
            "**2. Validate vs gold**: rank the ERAD gold set end-to-end "
            "and score the leaderboard against the known tier order. Use "
            "as a regression test after any prompt or model change."
        )
        _show_saved("validate")
        col_a, col_b = st.columns(2)
        with col_a:
            va_cycles = st.slider("Cycles", 2, 10, 6, key="va_cycles")
        with col_b:
            va_npc = st.slider("Matches per cycle", 4, 12, 8, key="va_npc")
        va_calls = va_cycles * va_npc * 2            # fair judge = 2 calls/match
        va_cost = 0.001 * va_calls
        st.caption(f"~{va_calls} judge calls. Estimated spend: ~${va_cost:.2f}. "
                   f"Runtime: a few minutes.")
        if st.button("Run validate", type="primary",
                     disabled=not os.environ.get("ANTHROPIC_API_KEY"),
                     key="va_btn"):
            from benchmark.run_benchmark import (run_tournament, _fair_judge_fn,
                                                  CRITERIA)
            from benchmark.gold_set import GOLD, gold_hypotheses
            from benchmark.metrics import spearman, topk_jaccard
            hyps = gold_hypotheses()
            with st.spinner(f"Running {va_cycles}×{va_npc} matches with the "
                             "fair judge…"):
                run_tournament(hyps, _fair_judge_fn(),
                                cycles=int(va_cycles),
                                n_per_cycle=int(va_npc))
            # Score against gold
            gold_rank_of = {h.id: i for i, h in enumerate(hyps)}
            gold_scores = [-gold_rank_of[h.id] for h in hyps]
            elo_scores = [h.elo for h in hyps]
            rho = spearman(gold_scores, elo_scores)
            final = sorted(hyps, key=lambda h: -h.elo)
            gold_order = [h.id for h in sorted(hyps,
                                               key=lambda h: gold_rank_of[h.id])]
            final_order = [h.id for h in final]
            top1 = final_order[0] == gold_order[0]
            top3 = topk_jaccard(final_order, gold_order, 3)

            c1, c2, c3 = st.columns(3)
            c1.metric("spearman vs gold", f"{rho:+.2f}",
                      help="1.0 = matches gold tier order exactly.")
            c2.metric("top-1 correct", "yes" if top1 else "no",
                      help="Did a tier-A hypothesis finish #1?")
            c3.metric("top-3 overlap", f"{top3:.0%}")
            bench_results.save_run("validate", {
                "label": "Validate vs gold",
                "params": {"cycles": int(va_cycles), "matches/cycle": int(va_npc)},
                "metrics": {"spearman vs gold": f"{rho:+.2f}",
                            "top-1 correct": "yes" if top1 else "no",
                            "top-3 overlap": f"{top3:.0%}"},
            })

            import pandas as pd
            rows = []
            for rank, h in enumerate(final, 1):
                tier = GOLD[next(i for i, x in enumerate(hyps)
                                 if x.id == h.id)]["tier"]
                rows.append({
                    "rank": rank,
                    "tier": tier,
                    "Elo": round(h.elo, 0),
                    "statement": h.statement,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

    # ---- 3. Compare fair vs naive ----------------------------------------
    with st.container(border=True):
        st.markdown(
            "**3. Compare fair vs naive judge**: same gold set, ranked "
            "under both judges. Δ Spearman tells you whether the "
            "order-swap is actually buying accuracy."
        )
        _show_saved("compare_fair_naive")
        col_a, col_b = st.columns(2)
        with col_a:
            cm_cycles = st.slider("Cycles", 2, 10, 6, key="cm_cycles")
        with col_b:
            cm_npc = st.slider("Matches per cycle", 4, 12, 8, key="cm_npc")
        cm_calls = cm_cycles * cm_npc * 3            # naive 1× + fair 2× per match
        cm_cost = 0.001 * cm_calls
        st.caption(f"~{cm_calls} judge calls (naive once + fair twice per "
                   f"match). Estimated spend: ~${cm_cost:.2f}.")
        if st.button("Run compare", type="primary",
                     disabled=not os.environ.get("ANTHROPIC_API_KEY"),
                     key="cm_btn"):
            from benchmark.run_benchmark import (run_tournament,
                                                  _fair_judge_fn,
                                                  _naive_judge_fn)
            from benchmark.gold_set import gold_hypotheses
            from benchmark.metrics import spearman

            def score(hyps):
                gold_rank_of = {h.id: i for i, h in enumerate(hyps)}
                return spearman([-gold_rank_of[h.id] for h in hyps],
                                [h.elo for h in hyps])

            with st.spinner("Running naive judge…"):
                h1 = gold_hypotheses()
                run_tournament(h1, _naive_judge_fn(),
                                cycles=int(cm_cycles),
                                n_per_cycle=int(cm_npc))
                rho_naive = score(h1)
            with st.spinner("Running fair judge…"):
                h2 = gold_hypotheses()
                run_tournament(h2, _fair_judge_fn(),
                                cycles=int(cm_cycles),
                                n_per_cycle=int(cm_npc))
                rho_fair = score(h2)

            c1, c2, c3 = st.columns(3)
            c1.metric("naive judge: spearman", f"{rho_naive:+.2f}")
            c2.metric("fair judge: spearman", f"{rho_fair:+.2f}")
            c3.metric("Δ (fair − naive)", f"{rho_fair - rho_naive:+.2f}",
                      delta=f"{rho_fair - rho_naive:+.2f}")
            bench_results.save_run("compare_fair_naive", {
                "label": "Fair vs naive judge",
                "params": {"cycles": int(cm_cycles), "matches/cycle": int(cm_npc)},
                "metrics": {"naive: spearman": f"{rho_naive:+.2f}",
                            "fair: spearman": f"{rho_fair:+.2f}",
                            "Δ (fair − naive)": f"{rho_fair - rho_naive:+.2f}"},
            })
            if rho_fair > rho_naive:
                st.success("Fair judge ranks closer to the gold tier order.")
            elif rho_fair < rho_naive:
                st.warning(
                    "Fair judge underperformed in this run. Re-run a few "
                    "times, because a single comparison is one noisy sample. "
                    "Trust the median of 3+ runs."
                )

    # ---- 4. Ontology grounding (structured-knowledge layer) --------------
    with st.container(border=True):
        st.markdown(
            "**4. Ontology grounding**: the fair judge ranks the same gold "
            "set with vs without a canonical 'known-biology' block "
            "(genes / diseases / pathways resolved via "
            "[OntoMCP](https://github.com/jeanlouishoneine-tech/OntoMCP)) "
            "injected into the prompt. Δ Spearman tells you whether grounding "
            "in structured knowledge actually beats reading the text alone."
        )
        _show_saved("compare_ontology")

        from co_scientist.ontology import ontomcp_available, ONTOMCP_API_URL
        onto_up = ontomcp_available()
        if onto_up:
            st.caption(f"✓ OntoMCP reachable at {ONTOMCP_API_URL}")
        else:
            st.info(
                "This comparison needs a local **OntoMCP** server, which isn't "
                "part of the hosted demo. To run it yourself: clone and start "
                "OntoMCP in a separate terminal, add your Anthropic key in the "
                "sidebar, then reload.\n\n"
                "```\ngit clone https://github.com/jeanlouishoneine-tech/"
                "OntoMCP.git && cd OntoMCP\nuv sync && uv run ontomcp-api\n```\n"
                f"(Set `ONTOMCP_API_URL` if it isn't on {ONTOMCP_API_URL}.)"
            )
        col_a, col_b = st.columns(2)
        with col_a:
            on_cycles = st.slider("Cycles", 2, 10, 6, key="on_cycles")
        with col_b:
            on_npc = st.slider("Matches per cycle", 4, 12, 8, key="on_npc")
        on_calls = on_cycles * on_npc * 4        # fair judge twice, both arms
        st.caption(f"~{on_calls} judge calls (fair judge twice per match, "
                   f"both arms). Estimated spend: ~${0.001 * on_calls:.2f}. "
                   "OntoMCP lookups are local and free.")
        if st.button("Run ontology compare", type="primary",
                     disabled=not (os.environ.get("ANTHROPIC_API_KEY")
                                   and onto_up),
                     key="on_btn"):
            from benchmark.run_benchmark import run_tournament, _fair_judge_fn
            from benchmark.gold_set import gold_hypotheses
            from benchmark.metrics import spearman

            def score(hyps):
                gold_rank_of = {h.id: i for i, h in enumerate(hyps)}
                return spearman([-gold_rank_of[h.id] for h in hyps],
                                [h.elo for h in hyps])

            with st.spinner("Running fair judge, grounding OFF…"):
                h1 = gold_hypotheses()
                run_tournament(h1, _fair_judge_fn(ontology=False),
                                cycles=int(on_cycles), n_per_cycle=int(on_npc))
                rho_base = score(h1)
            with st.spinner("Running fair judge, grounding ON…"):
                h2 = gold_hypotheses()
                run_tournament(h2, _fair_judge_fn(ontology=True),
                                cycles=int(on_cycles), n_per_cycle=int(on_npc))
                rho_onto = score(h2)

            c1, c2, c3 = st.columns(3)
            c1.metric("grounding OFF: spearman", f"{rho_base:+.2f}")
            c2.metric("grounding ON: spearman", f"{rho_onto:+.2f}")
            c3.metric("Δ (ON − OFF)", f"{rho_onto - rho_base:+.2f}",
                      delta=f"{rho_onto - rho_base:+.2f}")
            bench_results.save_run("compare_ontology", {
                "label": "Ontology grounding (fair judge, OFF vs ON)",
                "params": {"cycles": int(on_cycles),
                           "matches/cycle": int(on_npc)},
                "metrics": {"grounding OFF. Spearman": f"{rho_base:+.2f}",
                            "grounding ON: spearman": f"{rho_onto:+.2f}",
                            "Δ (ON − OFF)": f"{rho_onto - rho_base:+.2f}"},
            })
            if rho_onto > rho_base:
                st.success("Ontology grounding ranks closer to the gold "
                           "tier order.")
            elif rho_onto < rho_base:
                st.warning(
                    "Grounding underperformed in this run. One comparison "
                    "is one noisy sample. Trust the median of 3+ runs."
                )
            else:
                st.info("No difference this run. Re-run a few times before "
                        "concluding.")

    # ---- 5. Ontology discrimination (traps vs novelty) -------------------
    with st.container(border=True):
        st.markdown(
            "**5. Ontology discrimination**: the honest version of #4. Ranks an "
            "adversarial set with three kinds of hypothesis (solid, fluent-but-"
            "**false** traps, and **novel**-but-true) and asks: does grounding "
            "sink the traps **without** punishing the novel ideas? "
            "*Trap demotion* should be positive; *novelty penalty* should be ~0."
        )
        _show_saved("discrimination")
        if not IS_HOSTED:
            d_cols = st.columns(2)
            with d_cols[0]:
                ad_cycles = st.slider("Cycles", 2, 10, 6, key="ad_cycles")
            with d_cols[1]:
                ad_npc = st.slider("Matches per cycle", 4, 12, 8, key="ad_npc")
            st.caption(f"~{ad_cycles * ad_npc * 4} judge calls (fair judge twice, "
                       f"both arms). Needs OntoMCP running.")
            if st.button("Run discrimination test", type="primary",
                         disabled=not (os.environ.get("ANTHROPIC_API_KEY")
                                       and onto_up),
                         key="ad_btn"):
                from benchmark.run_benchmark import run_tournament, _fair_judge_fn
                from benchmark.gold_set_adversarial import adversarial_hypotheses
                from benchmark.run_adversarial import summarize
                with st.spinner("Ranking adversarial set, grounding OFF…"):
                    off = adversarial_hypotheses()
                    run_tournament(off, _fair_judge_fn(ontology=False),
                                   int(ad_cycles), int(ad_npc))
                with st.spinner("Ranking adversarial set, grounding ON…"):
                    on = adversarial_hypotheses()
                    run_tournament(on, _fair_judge_fn(ontology=True),
                                   int(ad_cycles), int(ad_npc))
                res = summarize(off, on)
                c1, c2, c3 = st.columns(3)
                c1.metric("trap demotion", f"{res['trap_demotion']:+.1f}",
                          help="Want > 0. Grounding sinks the false traps.")
                c2.metric("novelty penalty", f"{res['novelty_penalty']:+.1f}",
                          help="Want ~0: grounding leaves novel ideas alone.")
                c3.metric("spearman ON", f"{res['spearman_on']:+.2f}",
                          delta=f"{res['spearman_on'] - res['spearman_off']:+.2f}")
                bench_results.save_run("discrimination", {
                    "label": "Ontology discrimination",
                    "params": {"cycles": int(ad_cycles), "matches/cycle": int(ad_npc)},
                    "metrics": {"trap demotion": f"{res['trap_demotion']:+.1f}",
                                "novelty penalty": f"{res['novelty_penalty']:+.1f}",
                                "spearman ON": f"{res['spearman_on']:+.2f}"},
                })
                if res["trap_demotion"] > 0 and res["novelty_penalty"] <= 0.75:
                    st.success("Grounding caught the traps without punishing novelty.")
                elif res["trap_demotion"] <= 0:
                    st.warning("Grounding didn't demote the traps. Coverage / "
                               "placement issue. (One run is one noisy sample.)")
                else:
                    st.warning("Traps sank but novel ideas dropped too. The "
                               "consensus-filter risk. Try grounding in review only.")

    # ---- 6. Alias / duplicate detection ----------------------------------
    with st.container(border=True):
        st.markdown(
            "**6. Alias / duplicate detection**: where the ontology beats text "
            "similarity. Two hypotheses can be the *same idea* worded differently "
            "(\"multiple myeloma\" vs \"plasma cell myeloma\"). The metric is "
            "**separation**: how much higher a method scores true duplicates than "
            "unrelated pairs. No API key needed. Just embeddings + OntoMCP lookups."
        )
        _show_saved("alias_dedup")
        if st.button("Run alias-dedup", key="al_btn"):
            from benchmark.alias_dedup import (summarize, _verdict,
                                               _build_rows_live, _build_rows_demo)
            from co_scientist.ontology import ontomcp_available
            live = ontomcp_available() and not IS_HOSTED
            rows = _build_rows_live() if live else _build_rows_demo()
            st.caption("Source: live OntoMCP" if live
                       else "synthetic demo data (OntoMCP not reachable / hosted)")
            res = summarize(rows)
            c1, c2 = st.columns(2)
            c1.metric("ontology separation", f"{res['ontology_separation']:+.2f}",
                      help="same − different. Higher = better at spotting duplicates.")
            c2.metric("embedding separation", f"{res['embedding_separation']:+.2f}")
            import pandas as pd
            df = pd.DataFrame(
                [{"pair": r["kind"], "label": r["label"],
                  "ontology": round(r["onto"], 2), "embedding": round(r["emb"], 2),
                  "note": ("⚠ not unified" if r.get("onto_na") else "")}
                 for r in rows])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(_verdict(res))
            bench_results.save_run("alias_dedup", {
                "label": "Alias / duplicate detection",
                "params": {"pairs": len(rows)},
                "metrics": {"ontology separation": f"{res['ontology_separation']:+.2f}",
                            "embedding separation": f"{res['embedding_separation']:+.2f}"},
            })

    st.divider()
    st.caption(
        "Full plan and recommended sequence of work: "
        "[benchmark/BENCHMARKING_PLAN.md]"
        "(https://github.com/nataliegits/Benchmate/blob/main/benchmark/"
        "BENCHMARKING_PLAN.md). Gold set lives in `benchmark/gold_set.py` "
        "(currently ERAD / bortezomib-resistant multiple myeloma). "
        "The same benchmarks also run as CLI commands "
        "(`python -m benchmark.run_benchmark <subcommand>`)."
    )

# ── Tab 3. Cross-check your hypotheses ──────────────────────
with tab3:
    step_intro("your hypothesis has a rank: now find out whether five independent models back it, and whether the ranking itself is reliable.", "4 of 5", "Run experiment")
    from benchmark import results as bench_results
    from benchmark.elo_vs_variant_score import correlate, _load, _demo
    IS_HOSTED = bool(os.environ.get("BENCHMATE_HOSTED"))

    def _show_saved(name: str) -> None:
        run = bench_results.latest(name)
        if not run:
            if IS_HOSTED:
                st.caption("No saved run yet. Run this locally to populate the demo.")
            return
        p = run.get("params", {})
        bits = ", ".join(f"{k}={v}" for k, v in p.items())
        st.caption(f"📊 Saved run{(' · ' + bits) if bits else ''} · "
                   f"captured {run.get('captured', '?')}")
        metrics = run.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k, v)

    def _show_correlation(elo, score, labels, src, save_key, score_col):
        import pandas as pd
        res = correlate(elo, score)
        st.caption(f"Source: {src}")
        if res["spearman"] is None:
            st.info(res["note"]); return
        st.metric("Spearman(Elo, predictor)", f"{res['spearman']:+.2f}")
        st.caption(res["verdict"])
        df = pd.DataFrame(sorted(zip(labels, elo, score), key=lambda t: -t[1]),
                          columns=["hypothesis", "Elo", score_col])
        st.dataframe(df, use_container_width=True, hide_index=True)
        bench_results.save_run(save_key, {
            "label": save_key, "params": {"n": res["n"]},
            "metrics": {"Spearman(Elo, predictor)": f"{res['spearman']:+.2f}"}})

    st.markdown("### Cross-check with other models")
    st.markdown(
        "The Elo leaderboard is the LLM judge's opinion. Before trusting it to pick "
        "a wet-lab candidate, cross-check it against **independent quantitative "
        "models**: each scoring a hypothesis from a completely different angle. "
        "Low correlation = a flag. This is the *panel of judges*."
    )

    _sh_xc = st.session_state.get("sh_crosscheck")
    if _sh_xc:
        lines = "\n".join(
            f"- **{str(x.get('target', '?')).upper()}** → "
            f"*{x.get('model', '?')}* ({x.get('reason', '')})"
            for x in _sh_xc
        )
        st.info("From your Start here plan, check these:\n\n" + lines)

    # ---- Score one real hypothesis, live ------------------------------------
    # One box per model. Each shows what it pulled out of your hypothesis, what
    # question it answers, and its own run button. Because a single combined
    # button made four of the five models look broken when they were simply
    # answering a different question or waiting on a key.
    from co_scientist import crosscheck as _xc
    from co_scientist import hypothesis_scan as _hs
    from co_scientist import target_scorer
    from benchmark import live_scores as _live

    # Version skew shows up here first, because this tab uses the newest code.
    # Heal it if we can; if not, say what to click instead of crashing with a
    # redacted AttributeError.
    _skew = _heal_stale_modules()
    if _skew:
        st.error(
            "This server is running an older copy of Benchmate's modules than "
            "the interface expects, so some models can't load.\n\n"
            "**Fix:** on Streamlit Cloud, open *Manage app* (lower right) → "
            "**Reboot app**. Locally, stop and restart `streamlit run`.\n\n"
            + "\n".join(f"- `{s}`" for s in _skew))
        st.stop()

    st.markdown("#### Score a hypothesis")
    st.caption("Pick a hypothesis, then run whichever models apply. Each box "
               "says what it needs and what it found.")

    _hyps: list[str] = []
    _sf = REPO_ROOT / "state.json"
    if _sf.exists():
        try:
            _hyps = [h.get("statement", "") for h in sorted(
                json.loads(_sf.read_text()).get("hypotheses", []),
                key=lambda h: h.get("elo", 0), reverse=True)[:5]
                if h.get("statement")]
        except Exception:
            pass

    if _hyps:
        _pick = st.selectbox(
            "From your leaderboard", _hyps,
            format_func=lambda s: (s[:110] + "…") if len(s) > 110 else s,
            key="xc_pick")
        if st.session_state.get("_xc_last_pick") != _pick:
            st.session_state["xc_text"] = _pick
            st.session_state["_xc_last_pick"] = _pick
            # a new hypothesis invalidates every model's result
            for _k in ("xc_ot", "xc_dm", "xc_am"):
                st.session_state.pop(_k, None)
    else:
        st.caption("No leaderboard yet. Run **Generate & rank** first, or paste "
                   "a hypothesis below.")
    st.session_state.setdefault("xc_text", "")
    st.text_area("Hypothesis", key="xc_text", height=80)

    _htext = st.session_state.get("xc_text", "")
    _scan = _hs.scan(_htext) if _htext.strip() else {
        "genes": [], "variants": [], "scoreable_variants": [], "notes": [],
        "validated": True}
    _genes = _scan["genes"]

    if _htext.strip():
        st.markdown("**Pulled from this hypothesis:** "
                    + (", ".join(f"`{g}`" for g in _genes) or "_no genes found_")
                    + (("  ·  variants: "
                        + ", ".join(f"`{v['raw']}`" for v in _scan["variants"]))
                       if _scan["variants"] else ""))
        if not _scan.get("validated", True):
            st.caption("⚠ Symbols matched by pattern only. The gene validator "
                       "was unreachable, so one may not be a real gene.")
        if not _genes:
            st.caption("Name the gene explicitly (e.g. `SEL1L`, not "
                       "\"the ERAD receptor\") and these models can look it up.")

    def _q(template: str) -> str:
        """Put the hypothesis's own genes into a model's question.

        "Do cancer cells need UBE2J1 to survive?" lands; "do cancer cells need
        this gene to survive?" makes you do the substitution in your head.
        """
        if not _genes:
            return template.replace("{g}", "this gene")
        shown = ", ".join(_genes[:3])
        if len(_genes) > 3:
            shown += f" +{len(_genes) - 3} more"
        return template.replace("{g}", shown)

    def _show_work(exps, label):
        """Render the real steps a model took, not a description of them."""
        if not exps:
            return
        with st.expander(f"What Benchmate did to get {label}"):
            for e in exps:
                if e.get("gene"):
                    st.markdown(f"**{e['gene']}**")
                for _s in e.get("steps", []):
                    st.markdown(f"{_s}")
                if e.get("variants"):
                    import pandas as pd
                    st.dataframe(pd.DataFrame([
                        {"Variant": v["variant"], "ClinVar": v["clinvar"],
                         "AlphaMissense": v["alphamissense"]}
                        for v in e["variants"]]),
                        use_container_width=True, hide_index=True)
                st.divider()
            _d = exps[0].get("detail")
            if _d:
                st.caption(_d)

    def _score_table(rows):
        import pandas as pd
        st.dataframe(
            pd.DataFrame([{"Target": r["target"], "Score": r["score"],
                           "What it means": r["reading"]} for r in rows]),
            use_container_width=True, hide_index=True)

    # ---------------- Open Targets ----------------
    with st.container(border=True):
        st.markdown("**Open Targets**: *"
                    + _q("is {g} actually linked to this disease?") + "*")
        st.caption("Public API · nothing to install · genetics, literature and "
                   "known drugs, aggregated into one 0-1 association score.")
        _c1, _c2 = st.columns([2, 1])
        _dis = _c1.text_input("Disease", value=_xc.DEFAULT_DISEASE,
                              key="xc_disease", label_visibility="collapsed",
                              placeholder="disease, e.g. multiple myeloma")
        with _c2:
            _run_ot = st.button("Run Open Targets", key="xc_ot_btn",
                                disabled=not _genes, use_container_width=True)
        if _run_ot:
            with st.spinner("Asking Open Targets…"):
                _rows, _probs, _exps = [], [], []
                for _g in _genes:
                    _e = _xc.explain_opentargets(_g, _dis)
                    _e["gene"] = _g
                    _exps.append(_e)
                    _v = _e["score"]
                    if _v is None:
                        _probs.append(f"{_g}: didn't resolve, or the API errored.")
                    else:
                        _rows.append({"target": _g, "score": round(_v, 3),
                                      "reading": _xc.read_opentargets(_v)})
                _elo = _live.elo_for_statement(_htext)
                for _row in _rows:
                    _live.record("Open Targets", _htext[:120], _elo,
                                 _row["score"], _row["target"])
                _tr("crosscheck", f"Open Targets scored {len(_rows)} gene(s)",
                    detail=_htext[:200],
                    inputs={"genes": ", ".join(_genes), "disease": _dis},
                    outputs={r["target"]: f"{r['score']} ({r['reading']})"
                             for r in _rows})
                st.session_state["xc_ot"] = {"rows": _rows, "problems": _probs,
                                             "exps": _exps}
        _r = st.session_state.get("xc_ot")
        if _r:
            if _r["rows"]:
                _score_table(_r["rows"])
                st.caption("0-1, higher = stronger evidence the gene is involved "
                           "in this disease. 0.0 means resolved cleanly with "
                           "nothing on record.")
            _show_work(_r.get("exps"), "this score")
            for _p in _r["problems"]:
                st.caption(f"⚠ {_p}")
        elif not _genes:
            st.caption("Waiting on a gene from the hypothesis above.")

    # ---------------- DepMap ----------------
    with st.container(border=True):
        st.markdown("**DepMap**: *"
                    + _q("do cancer cells actually need {g} to survive?") + "*")
        _dm_ok = target_scorer.depmap_available()
        if _dm_ok:
            _dm_src = target_scorer.depmap_source()
            _dm_lin = target_scorer.depmap_lineage_in_use()
            st.caption(
                ("Full CRISPR knockout matrix on this machine · "
                 if _dm_src == "full" else
                 "Precomputed gene-effect summary that ships with Benchmate · ")
                + (f"averaged over {_dm_lin} cancer cell lines"
                   if _dm_lin == "all" else
                   f"restricted to {_dm_lin} cell lines"))
            if st.button("Run DepMap", key="xc_dm_btn", disabled=not _genes):
                with st.spinner("Reading the CRISPR matrix…"):
                    _rows, _probs, _exps = [], [], []
                    for _g in _genes:
                        _e = _xc.explain_depmap(_g)
                        _e["gene"] = _g
                        _exps.append(_e)
                        _v = _e["score"]
                        if _v is None:
                            _probs.append(f"{_g}: not in the CRISPR data.")
                        else:
                            _rows.append({"target": _g, "score": round(_v, 3),
                                          "reading": _xc.read_depmap(_v)})
                    _elo = _live.elo_for_statement(_htext)
                    for _row in _rows:
                        _live.record("DepMap", _htext[:120], _elo,
                                     _row["score"], _row["target"])
                    _tr("crosscheck", f"DepMap scored {len(_rows)} gene(s)",
                        detail=_htext[:200],
                        inputs={"genes": ", ".join(_genes)},
                        outputs={r["target"]: f"{r['score']} ({r['reading']})"
                                 for r in _rows})
                    st.session_state["xc_dm"] = {"rows": _rows,
                                                 "problems": _probs, "exps": _exps}
            _r = st.session_state.get("xc_dm")
            if _r:
                if _r["rows"]:
                    _score_table(_r["rows"])
                    st.caption("Higher = more essential. Above ~1.0 the cells "
                               "die without it; near 0 it's dispensable.")
                _show_work(_r.get("exps"), "this score")
                for _p in _r["problems"]:
                    st.caption(f"⚠ {_p}")
            elif not _genes:
                st.caption("Waiting on a gene from the hypothesis above.")
        else:
            # getattr, not attribute access: this branch runs precisely when
            # something is missing, so it must not itself depend on a new
            # attribute existing.
            _sum_path = getattr(target_scorer, "DEPMAP_SUMMARY",
                                target_scorer.DEPMAP_CSV.parent
                                / "gene_effect_summary.csv")
            st.warning(
                "No DepMap data found. This shouldn't happen, since a "
                "precomputed summary ships with Benchmate. Expected it at "
                f"`{_sum_path}`.")
            st.caption("To rebuild it: download CRISPRGeneEffect.csv from "
                       "[depmap.org](https://depmap.org/portal/data_page/) and "
                       "run `python -m benchmark.build_depmap_summary`.")

    # ---------------- AlphaMissense ----------------
    with st.container(border=True):
        st.markdown("**AlphaMissense**: *"
                    + _q("would coding changes in {g} be damaging?") + "*")
        st.caption("Free Ensembl VEP API · nothing to install. It scores one base "
                   "change at a time, so for a gene we score that gene's known "
                   "pathogenic missense variants from ClinVar and average them. "
                   "Real coordinates only. Nothing is generated.")
        _sv = _scan["scoreable_variants"]
        if _sv:
            st.caption(f"This hypothesis names coordinates ({_sv[0]['raw']}), so "
                       f"that exact variant gets scored instead.")
        if st.button("Run AlphaMissense", key="xc_am_btn",
                     disabled=not (_genes or _sv)):
            with st.spinner("Fetching ClinVar variants and scoring…"):
                _rows, _probs, _exps = [], [], []
                if _sv:
                    for _v in _sv:
                        _s = target_scorer.alphamissense_score(
                            _v["chrom"], _v["pos"], _v["ref"], _v["alt"])
                        if _s is None:
                            _probs.append(f"{_v['raw']}: no score. It may not be "
                                          f"a missense change.")
                        else:
                            _rows.append({"target": _v["raw"], "score": round(_s, 3),
                                          "reading": _xc.read_alphamissense(_s)})
                else:
                    for _g in _genes:
                        _e = _xc.explain_missense(_g)
                        _e["gene"] = _g
                        _exps.append(_e)
                        _m, _n, _why = _e["score"], _e["n"], _e["why"]
                        if _m is None:
                            _probs.append(_why)
                        else:
                            _rows.append({
                                "target": f"{_g} (gene-level)",
                                "score": round(_m, 3),
                                "reading": (f"{_xc.read_alphamissense(_m)}: mean "
                                            f"over {_n} ClinVar variants")})
                _elo = _live.elo_for_statement(_htext)
                for _row in _rows:
                    _live.record("AlphaMissense", _htext[:120], _elo,
                                 _row["score"], _row["target"])
                _tr("crosscheck", f"AlphaMissense scored {len(_rows)} target(s)",
                    detail=_htext[:200],
                    outputs={r["target"]: f"{r['score']} ({r['reading']})"
                             for r in _rows})
                st.session_state["xc_am"] = {"rows": _rows, "problems": _probs,
                                             "exps": _exps}
        _r = st.session_state.get("xc_am")
        if _r:
            if _r["rows"]:
                _score_table(_r["rows"])
                st.caption("0-1. Above 0.564 is AlphaMissense's "
                           "likely-pathogenic threshold; below 0.34 is likely "
                           "benign.")
            _show_work(_r.get("exps"), "this score")
            for _p in _r["problems"]:
                st.caption(f"⚠ {_p}")
        elif not (_genes or _sv):
            st.caption("Waiting on a gene from the hypothesis above.")

    # ---------------- AlphaGenome ----------------
    with st.container(border=True):
        st.markdown("**AlphaGenome**: *"
                    + _q("would a regulatory variant change {g} expression?") + "*")
        st.caption("Needs a free API key and Python 3.10+, so it runs in Colab. "
                   "Benchmate writes the notebook against real GTEx eQTLs for "
                   "your genes; you run it and bring back the scores.")
        st.markdown("Get a free key from [DeepMind]"
                    "(https://deepmind.google.com/science/alphagenome/) "
                    "(non-commercial use): the notebook prompts you for it.")

        _ag_genes = list(_genes)
        _src_label = "the hypothesis above"
        if not _ag_genes:
            for _x in (st.session_state.get("sh_crosscheck") or []):
                _t = str(_x.get("target", "")).strip().upper()
                if _t and _t not in _ag_genes:
                    _ag_genes.append(_t)
            if _ag_genes:
                _src_label = "your Start here plan"
        if not _ag_genes:
            try:
                _ag_genes = list(available_geneformer_genes())[:6]
                _src_label = "your perturbation cache"
            except Exception:
                _src_label = "nothing yet"

        st.caption(f"Genes from **{_src_label}**: edit freely.")
        _ag_txt = st.text_input("Genes to score (comma-separated)",
                                value=", ".join(_ag_genes), key="ag_nb_genes")
        _ag_list = [g.strip().upper() for g in _ag_txt.split(",") if g.strip()]

        if st.button("Write the Colab notebook", key="ag_nb_btn",
                     type="primary"):
            st.session_state.pop("ag_colab_url", None)
            try:
                from ui.alphagenome_nb import generate_alphagenome_notebook
                with st.spinner("Looking up real eQTLs in GTEx…"):
                    _p, _n = generate_alphagenome_notebook(genes=_ag_list)
                st.session_state["ag_nb_path"] = str(_p)
                st.session_state["ag_nb_n"] = _n
                _tr("notebook", f"Wrote an AlphaGenome notebook ({_n} variants)",
                    detail="Real GTEx eQTLs for " + (", ".join(_ag_list)
                                                     or "the ERAD benchmark set"),
                    files=[str(_p)])
                # Straight into Colab rather than download-then-upload: push the
                # notebook to a gist and hand back the Colab URL. Same handoff
                # the Geneformer tab uses.
                if gh_available():
                    with st.spinner("Opening it in Colab…"):
                        _h = handoff(_p, description=(
                            f"Benchmate AlphaGenome scoring: "
                            f"{', '.join(_ag_list) or 'ERAD benchmark set'}"))
                    st.session_state["ag_colab_url"] = _h["colab_url"]
            except GhUnavailable:
                pass          # no gist creds. The download fallback still works
            except Exception as ex:
                st.session_state.pop("ag_nb_path", None)
                st.error(str(ex))
        _agp = st.session_state.get("ag_nb_path")
        if _agp and Path(_agp).exists():
            _pth = Path(_agp)
            st.success(f"`{_pth.name}`: {st.session_state.get('ag_nb_n', '?')} "
                       f"variants, each a real GTEx eQTL.")
            _cu = st.session_state.get("ag_colab_url")
            if _cu:
                st.link_button("▶ Open in Colab", _cu, type="primary",
                               use_container_width=True)
                st.caption("Opens ready to run. No download, no upload. Run top "
                           "to bottom and it produces "
                           "`alphagenome_scores.json`.")
            else:
                st.download_button("Download the notebook", _pth.read_bytes(),
                                   file_name=_pth.name, key="ag_nb_dl",
                                   use_container_width=True)
                st.caption(
                    "One-click Colab needs gist access. Set a `GITHUB_TOKEN` "
                    "(Streamlit Cloud: *Manage app → Settings → Secrets*) with "
                    "the `gist` scope, and this button becomes **Open in "
                    "Colab**. Until then: download, then upload to Colab.")
            st.caption("Bring the scores back to the AlphaGenome panel under "
                       "**Calibration** below.")
        if not _ag_list:
            st.caption("With no genes it falls back to the built-in ERAD "
                       "benchmark set: right for calibration, but it won't "
                       "reflect your question.")

    # ---------------- Boltz ----------------
    with st.container(border=True):
        from co_scientist import boltz_scorer as _bz

        st.markdown("**Boltz**: *does the drug actually bind the target?*")
        st.caption("Co-folds a protein with a small molecule and reports a "
                   "binding confidence (0-1). Unlike the others it needs two "
                   "specific things a sentence doesn't carry: the protein's "
                   "amino-acid sequence, and the drug as SMILES.")

        # Session-only key. Never written to disk or Streamlit secrets. A paid
        # key in shared secrets would bill you for every visitor.
        _bk = st.text_input(
            "Boltz API key", type="password", key="bz_key",
            help="Kept in this session only. Not saved, not shared.",
            placeholder="paste your key from api.boltz.bio")
        if _bk:
            _bz.set_api_key(_bk)

        if not _bk:
            st.markdown("No key yet. Sign up at [api.boltz.bio]"
                        "(https://api.boltz.bio/console/signup) (launch credits "
                        "available) and paste the key above. It's the only model "
                        "here that costs money.")
        elif not _bz.sdk_installed():
            st.warning("Key accepted, but the `boltz-api` SDK isn't installed "
                       "in this environment. Add `boltz-api` to "
                       "`requirements.txt` and reboot the app.")
        else:
            st.success("Key set for this session.")
            _bc1, _bc2 = st.columns(2)
            _prot_gene = _bc1.text_input(
                "Protein (gene symbol)",
                value=(_genes[0] if _genes else ""), key="bz_gene",
                help="Benchmate fetches the canonical sequence from UniProt: "
                     "sequences are never made up.")
            # Nobody knows SMILES off-hand, and the Run button being greyed out
            # with no explanation was the whole blocker. So: type a drug name,
            # Benchmate resolves the structure from PubChem.
            _drug = _bc2.text_input(
                "Compound name", key="bz_drug",
                placeholder="e.g. bortezomib",
                help="Benchmate looks the structure up in PubChem. It never "
                     "makes a SMILES string up.")
            if _drug and st.session_state.get("_bz_drug_done") != _drug:
                from co_scientist.pubchem import smiles_for
                with st.spinner(f"Looking up {_drug} in PubChem…"):
                    _smi, _cid, _note = smiles_for(_drug)
                st.session_state["_bz_drug_done"] = _drug
                st.session_state["bz_smiles"] = _smi or ""
                st.session_state["_bz_cid"] = _cid
                st.session_state["_bz_note"] = _note

            _lig = st.text_input(
                "Ligand as SMILES", key="bz_smiles",
                placeholder="filled in from PubChem, or paste your own",
                help="A wrong SMILES silently scores a different molecule, so "
                     "check it matches the compound you mean.")
            if st.session_state.get("_bz_note"):
                st.warning(st.session_state["_bz_note"])
            elif st.session_state.get("_bz_cid"):
                from co_scientist.pubchem import pubchem_url
                _cid = st.session_state["_bz_cid"]
                st.caption(f"Structure from PubChem CID "
                           f"[{_cid}]({pubchem_url(_cid)}): confirm it's the "
                           f"compound you meant before spending a run.")

            if not (_prot_gene and _lig):
                st.caption("Needs both a gene symbol and a ligand. Type a "
                           "compound name above and the SMILES fills itself in.")
            if st.button("Run Boltz", key="bz_run",
                         disabled=not (_prot_gene and _lig), type="primary"):
                _seq, _acc, _err = None, None, None
                try:
                    from benchmark.fetch_uniprot import uniprot_sequence
                    with st.spinner(f"Fetching {_prot_gene} sequence from UniProt…"):
                        _seq, _acc = uniprot_sequence(_prot_gene)
                except Exception as e:
                    _err = f"UniProt lookup failed: {e}"
                if not _seq:
                    st.error(_err or f"No UniProt sequence found for "
                                     f"{_prot_gene}. Check the gene symbol: "
                                     f"Benchmate won't invent a sequence.")
                else:
                    st.caption(f"UniProt `{_acc}` · {len(_seq)} residues")
                    with st.spinner("Boltz is folding the complex. This takes "
                                    "a few minutes…"):
                        _sc = _bz.score_binding(_bz.BoltzTarget(
                            protein=_seq, ligand_smiles=_lig,
                            label=f"{_prot_gene}+ligand"))
                    _live.record("Boltz", _htext[:120],
                                 _live.elo_for_statement(_htext), _sc,
                                 _prot_gene)
                    st.session_state["xc_bz"] = {
                        "score": _sc, "gene": _prot_gene, "acc": _acc,
                        "n_res": len(_seq), "smiles": _lig}

            _r = st.session_state.get("xc_bz")
            if _r:
                if _r["score"] is None:
                    st.error("Boltz returned no score. The job may have failed "
                             "or timed out. The terminal running Streamlit "
                             "shows the API's reason.")
                else:
                    st.metric("Binding confidence", f"{_r['score']:.3f}")
                    st.caption("0-1. Above ~0.7 is Boltz's high-confidence "
                               "range; low values mean the model can't place "
                               "this ligand in the pocket, which is evidence "
                               "against a direct-binding hypothesis.")
                with st.expander("What Benchmate did to get this score"):
                    st.markdown(
                        f"- Fetched the canonical sequence for **{_r['gene']}** "
                        f"from **UniProt** (`{_r['acc']}`, {_r['n_res']} residues)\n"
                        f"- Submitted protein chain A + ligand chain B to the "
                        f"Boltz API as a `ligand_protein_binding` job "
                        f"(model `{_bz.BOLTZ_MODEL}`)\n"
                        f"- Polled until the job finished, then read "
                        f"`binding_metrics.binding_confidence`")
                    st.caption("Boltz is an AlphaFold3-class co-folding model: "
                               "it predicts the 3D structure of the complex and "
                               "how confident it is that the ligand binds. This "
                               "is a prediction, not a measurement.")

    # The five uploader panels below answer a DIFFERENT question from the
    # scoring panel above: not "is this hypothesis any good?" but "can I trust
    # the Elo ranking at all?" Folded away so the tab opens on the thing you
    # actually came to do.
    st.divider()
    with st.expander("Calibration: does the Elo ranking agree with these "
                     "models across a fixed gold set?"):
        st.caption("This measures the machinery, not your hypothesis. Each "
                   "panel takes a scores file and reports Spearman "
                   "correlation against Elo. Low correlation is a flag.")

        # The scores above and the files below are different data, and that
        # caught a real user out: score SEL1L live, open this, see SYVN1 and
        # PSMB5 from a benchmark run. Say so, and offer the live set instead.
        _lsum = _live.summary()
        _ready = [r for r in _lsum if r["ready"]]
        st.info(
            "**Two different datasets.** By default these panels read the "
            "**gold-set** files in `benchmark/`, built by benchmark scripts, so "
            "they will not contain the hypothesis you just scored above. "
            "Benchmate also records every live score against its hypothesis's "
            "Elo, which builds a calibration set from your own leaderboard.")
        if _lsum:
            st.caption("Recorded from your runs so far: " + " · ".join(
                f"**{r['model']}** {r['n']}"
                + ("" if r["ready"] else f"/{_live.MIN_POINTS} needed")
                for r in _lsum))
        else:
            st.caption("Nothing recorded live yet. Score a few different "
                       "hypotheses above and they will appear here.")
        _use_live = st.toggle(
            "Use my live scores instead of the gold set",
            value=bool(_ready), key="cal_use_live",
            help=f"Needs at least {_live.MIN_POINTS} different hypotheses "
                 f"scored by the same model for a correlation to mean anything.")
        # ----- AlphaGenome: regulatory / expression -----
        with st.container(border=True):
            st.markdown(
                "**AlphaGenome: regulatory effect** *(does a variant change expression?)*. "
                "Drop `variant_scores.json` (merged Elo + score) or a raw "
                "`alphagenome_scores.json` from Colab. See `benchmark/ALPHAGENOME_PLAN.md`."
            )
            _show_saved("elo_vs_predictor")
            with st.expander("How to produce real scores (AlphaGenome setup)"):
                st.markdown(
                    "1. Free key at [DeepMind](https://deepmind.google.com/science/alphagenome/) "
                    "→ *Get started* (sign in with a personal @gmail).\n"
                    "2. Generate the notebook above (**Generate an AlphaGenome "
                    "Colab notebook**) and run it on "
                    "[Colab](https://colab.research.google.com) → downloads "
                    "`alphagenome_scores.json`.\n"
                    "3. Put it in `benchmark/`, run `python -m benchmark.build_variant_scores`.\n"
                    "⚠️ Variant coordinates: run `python -m benchmark.fetch_eqtls` for real GTEx eQTLs."
                )
            up = st.file_uploader("Drop alphagenome_scores.json or variant_scores.json",
                                  type=["json"], key="ag_up")
            if st.button("Show AlphaGenome correlation", key="ag_btn"):
                import json
                elo = score = labels = None; src = ""
                if up is not None:
                    data = json.load(up)
                    if isinstance(data, list):
                        elo = [float(r["elo"]) for r in data]
                        score = [float(r["score"]) for r in data]
                        labels = [str(r.get("label", i)) for i, r in enumerate(data)]
                        src = "uploaded variant_scores.json"
                    elif isinstance(data, dict):
                        from benchmark.gold_set_variants import GOLD_VARIANTS
                        if os.environ.get("ANTHROPIC_API_KEY") and not IS_HOSTED:
                            from benchmark.build_variant_scores import _elo_by_label
                            with st.spinner("Ranking variant hypotheses for Elo…"):
                                em = _elo_by_label(6, 8)
                            trip = [(g["label"], em[g["label"]], float(data[g["label"]]))
                                    for g in GOLD_VARIANTS
                                    if g["label"] in data and g["label"] in em]
                            labels = [t[0] for t in trip]; elo = [t[1] for t in trip]
                            score = [t[2] for t in trip]
                            src = "uploaded AlphaGenome scores + freshly-ranked Elo"
                        else:
                            st.warning("Raw scores need an Elo column. Set your Anthropic "
                                       "key, run build_variant_scores locally, or upload "
                                       "the merged variant_scores.json.")
                else:
                    p = "benchmark/variant_scores.json"
                    if os.path.exists(p):
                        elo, score, labels = _load(p); src = f"`{p}`"
                    else:
                        elo, score, labels = _demo(); src = "synthetic demo data"
                if elo is not None:
                    _show_correlation(elo, score, labels, src,
                                      "elo_vs_predictor", "AlphaGenome score")

        # ----- Boltz: structure / binding -----
        with st.container(border=True):
            st.markdown(
                "**Boltz: structure & binding** *(does the drug actually bind the "
                "target?)*. The complement to AlphaGenome: it scores binding-style "
                "hypotheses. Drop a `boltz_scores.json` (merged Elo + score)."
            )
            _show_saved("elo_vs_boltz")
            with st.expander("How to produce Boltz scores"):
                st.markdown(
                    "1. Sign up + redeem **$100 credits (code BOLTZLAUNCH)** at "
                    "[api.boltz.bio](https://api.boltz.bio/console/signup), create an API "
                    "key, `export BOLTZ_API_KEY=...`.\n"
                    "2. For each binding hypothesis, score the protein+ligand with "
                    "`co_scientist/boltz_scorer.py` (`BoltzTarget` → `score_binding`).\n"
                    "3. Save `benchmark/boltz_scores.json` as "
                    "`[{label, elo, score}, ...]`.\n"
                    "⚠️ The Boltz API is new. Verify the endpoints in boltz_scorer.py "
                    "against the console's API reference."
                )
            upb = st.file_uploader("Drop boltz_scores.json", type=["json"], key="bz_up")
            if st.button("Show Boltz correlation", key="bz_btn"):
                import json
                elo = score = labels = None; src = ""
                if upb is not None:
                    data = json.load(upb)
                    if isinstance(data, list):
                        elo = [float(r["elo"]) for r in data]
                        score = [float(r["score"]) for r in data]
                        labels = [str(r.get("label", i)) for i, r in enumerate(data)]
                        src = "uploaded boltz_scores.json"
                    else:
                        st.warning("Upload a merged list of {label, elo, score}.")
                else:
                    p = "benchmark/boltz_scores.json"
                    if os.path.exists(p):
                        elo, score, labels = _load(p); src = f"`{p}`"
                    else:
                        st.info("No boltz_scores.json yet. See the setup steps above.")
                if elo is not None:
                    _show_correlation(elo, score, labels, src,
                                      "elo_vs_boltz", "Boltz score")

        # ----- generic gene/variant judges (Open Targets, DepMap, AlphaMissense) -----
        def _simple_panel(title, desc, default_path, save_key, up_key, btn_key,
                          score_col, setup_md=None):
            # honour the live-scores toggle: point at the file recorded from
            # this user's own runs rather than the shipped gold set
            _model_name = title.split(" ")[0] if title else ""
            if _use_live:
                _lp = _live.path_for(title.split(" -")[0].strip())
                if _lp.exists() and len(_live.load(title.split(" -")[0].strip())) >= 2:
                    default_path = str(_lp.relative_to(REPO_ROOT))
            with st.container(border=True):
                st.markdown(f"**{title}**: {desc}")
                if _use_live and "live_" in default_path:
                    st.caption(f"Reading **your live scores** "
                               f"(`{default_path}`), not the gold set.")
                _show_saved(save_key)
                if setup_md:
                    with st.expander("How to produce these scores"):
                        st.markdown(setup_md)
                up = st.file_uploader(f"Drop {default_path.split('/')[-1]}",
                                      type=["json"], key=up_key)
                if st.button(f"Show {title} correlation", key=btn_key):
                    import json
                    elo = score = labels = None; src = ""
                    if up is not None:
                        data = json.load(up)
                        if isinstance(data, list):
                            elo = [float(r["elo"]) for r in data]
                            score = [float(r["score"]) for r in data]
                            labels = [str(r.get("label", i)) for i, r in enumerate(data)]
                            src = f"uploaded {default_path.split('/')[-1]}"
                        else:
                            st.warning("Upload a merged list of {label, elo, score}.")
                    elif os.path.exists(default_path):
                        elo, score, labels = _load(default_path); src = f"`{default_path}`"
                    else:
                        st.info("No scores file yet. See the setup steps above.")
                    if elo is not None:
                        _show_correlation(elo, score, labels, src, save_key, score_col)

        _simple_panel(
            "Open Targets: disease association",
            "*is this gene genuinely linked to the disease?* (genetics + literature + drugs). "
            "Free, no key.",
            "benchmark/opentargets_scores.json", "elo_vs_opentargets",
            "ot_up", "ot_btn", "association",
            "Run `python -m benchmark.build_target_scores` (needs your Anthropic key; "
            "Open Targets is a free API). It writes `opentargets_scores.json`.")

        _simple_panel(
            "DepMap: gene dependency",
            "*is this gene actually essential in the disease's cancer cell lines?*",
            "benchmark/depmap_scores.json", "elo_vs_depmap",
            "dm_up", "dm_btn", "dependency",
            "Download `CRISPRGeneEffect.csv` from [depmap.org]"
            "(https://depmap.org/portal/data_page/) into `data/depmap/`, then run "
            "`python -m benchmark.build_target_scores`. **Optional:** also drop "
            "`Model.csv` there and the score is restricted to multiple-myeloma cell "
            "lines instead of pan-cancer.")

        _simple_panel(
            "AlphaMissense: variant pathogenicity",
            "*is a coding variant likely pathogenic?* (free, via Ensembl VEP). "
            "Scores missense variants. Supply real ones.",
            "benchmark/alphamissense_scores.json", "elo_vs_alphamissense",
            "am_up", "am_btn", "pathogenicity",
            "1. `python -m benchmark.fetch_clinvar` pulls **real** pathogenic + benign "
            "missense variants (GRCh38 coords) from ClinVar. 2. "
            "`python -m benchmark.build_missense_scores` ranks them by Elo, scores each "
            "with AlphaMissense (free Ensembl VEP), and writes "
            "`alphamissense_scores.json`: plus a pathogenic-vs-benign calibration check.")

    st.divider()
    with st.expander("Is the ranking itself reliable? (benchmark the tournament)"):
        st.caption("The panel above checks the *hypotheses*. This checks the "
                   "*machinery*: does the Elo tournament and the LLM judge produce "
                   "a ranking you can trust in the first place?")
        _benchmark_section()

    st.divider()
    benchmate_anchor("crosscheck", "*does the panel back the top idea?* · *which judge disagrees most, and why?*")

# ── Bench-assay panel (reused inside Experiment → Results) ───
def _bench_assay_panel():
    st.write(
        "Upload a run from the alamarBlue rig (columns `t_s, R, G, B, red_blue`). "
        "Benchmate turns the colour kinetics into a viability readout and files it "
        "as evidence: so the next run reasons over the bench result, not just the "
        "literature. This is the *test → learn* edge of the loop."
    )

    up = st.file_uploader("alamarBlue run CSV", type=["csv"], key="assay_up")
    col_a, col_b = st.columns(2)
    with col_a:
        a_hyp = st.text_input("Hypothesis label", value="SEL1L_kifunensine", key="a_hyp",
                              help="Ties this result to a hypothesis; agents match on it.")
        a_drug = st.text_input("Drug / treatment",
                               value="kifunensine (10 uM, 48h) + bortezomib", key="a_drug")
    with col_b:
        a_cell = st.text_input("Cell line", value="RPMI-8226, bortezomib-resistant", key="a_cell")
        a_ctrl = st.number_input("Vehicle-control Δ(red/blue) (optional)",
                                 value=0.24, min_value=0.0, step=0.01, key="a_ctrl",
                                 help="Supply the control's reduction Δ to express viability as % of control.")

    if st.button("Ingest assay → evidence", type="primary", disabled=up is None):
        import tempfile, pandas as pd
        tmp = Path(tempfile.mkdtemp()) / "run.csv"
        tmp.write_bytes(up.getvalue())
        try:
            _tr("bench", "Read an alamarBlue run from the rig")
            rec = assay.ingest(tmp, hypothesis=a_hyp.strip() or "assay_run",
                               drug=a_drug, cell=a_cell,
                               readout="alamarBlue red/blue reduction kinetics (TCS34725)",
                               control_delta=a_ctrl or None)
        except Exception as e:
            st.error(f"Could not read that CSV: {e}")
            st.stop()

        m = rec["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("baseline", m["baseline"])
        c2.metric("plateau", m["plateau"])
        c3.metric("Δ reduction", m["delta"])
        pct = rec["viability"].get("viability_pct_of_control")
        c4.metric("viability", f"{pct:.0f}%" if pct is not None else "-")

        rows = assay.read_run(tmp)
        df = pd.DataFrame(rows).set_index("t_s")[["red_blue"]]
        st.line_chart(df, height=240)

        verdict = rec["viability"]["verdict"]
        direction = rec["direction_for_benchmate"]
        (st.success if direction == "up-weight" else
         st.warning if direction == "down-weight" else st.info)(
            f"**{verdict}.** {rec['interpretation']} "
            f"Suggested action: **{direction}**.")
        st.caption(f"Saved to `data/rig/{a_hyp.strip()}_assay.json`: "
                   "the next Benchmate run will factor this in.")
        with st.expander("Evidence block the agents will read"):
            st.code(assay.summarize(rec))

    st.divider()
    on_record = assay.available_assays()
    if on_record:
        st.subheader("Bench results on record")
        for label in on_record:
            rec = assay.assay_evidence(label)
            if not rec:
                continue
            st.markdown(
                f"- **{label}**: {rec['drug']} on {rec['cell_line']}: "
                f"{rec['viability']['verdict']} "
                f"(*{rec['direction_for_benchmate']}*)"
            )
    else:
        st.caption("No bench results on record yet. Ingest a run above.")

# ── Tab 5. About (kept last) ────────────────────────────────
with tab5:
    step_intro("what Benchmate is, how the loop works, and where to read more.")
    st.header("About Benchmate")
    st.markdown(
        "Benchmate is a small, open AI co-scientist for biomedical hypothesis "
        "generation: an independent re-implementation of Google DeepMind's "
        "AI Co-Scientist. Seven agents propose hypotheses, critique each other, "
        "and run an Elo tournament; the winners are then cross-checked against a "
        "panel of independent models and, increasingly, real bench results."
    )
    st.markdown(
        "- **Live app:** [benchmate.streamlit.app](https://benchmate.streamlit.app)\n"
        "- **Code:** [github.com/nataliegits/Benchmate](https://github.com/nataliegits/Benchmate)\n"
        "- Built through the Worldwide Studios AI for Science fellowship."
    )

    st.subheader("Read on Substack")
    st.markdown(
        "The build, in order:\n\n"
        "1. [Building Benchmate, Part 1](https://benchpressed.substack.com/p/building-benchmate-part-1)\n"
        "2. [Building Benchmate, Part 2](https://benchpressed.substack.com/p/building-benchmate-part-2)\n"
        "3. [Can you trust an AI scientist's #1 idea?](https://benchpressed.substack.com/p/can-you-trust-an-ai-scientists-1)\n"
        "4. [Part 3. What is the AI actually judging?](https://benchpressed.substack.com/p/building-benchmate-part-3-what-is)\n"
        "5. [Part 4. A panel of judges](https://benchpressed.substack.com/p/building-benchmate-part-4-a-panel)\n"
        "6. [Part 5. Filling out the panel](https://benchpressed.substack.com/p/building-benchmate-part-5-filling)"
    )
    st.caption("New posts land at benchpressed.substack.com.")

    st.divider()
    benchmate_anchor("about", "*what can you do?* · *how does the loop work?* · *where should I start?*")

# ── Tab 4. Experiment (design → execute → results → inventory) ─
with tab4:
    step_intro("turns the winning hypothesis into a runnable assay, finds the reagents in your freezer, and reads the result back in.", "5 of 5", "loops back to step 3: the bench result re-ranks the ideas")
    st.header("Experiment: design, run, and learn")
    st.caption("Take a hypothesis to the bench and back: design the assay → find "
               "the reagents → run it → feed the result back to sharpen the idea.")

    DEFAULT_HYP = ("Inhibiting p97/VCP with CB-5083 re-imposes proteotoxic stress "
                   "and kills bortezomib-resistant multiple myeloma cells.")
    # let the Results step push a revised hypothesis back into Design (set before
    # the widget with key 'loop_hyp' is instantiated, so it takes effect)
    if "_pending_hyp" in st.session_state:
        st.session_state["loop_hyp"] = st.session_state.pop("_pending_hyp")
    st.session_state.setdefault("loop_hyp", DEFAULT_HYP)
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    d_tab, e_tab, r_tab, inv_tab = st.tabs([
        "1 · Design",
        "2 · Execute",
        "3 · Results & feedback",
        "Reagent inventory",
    ])

    # ---------- 1. Design the experiment ----------
    with d_tab:
        st.text_area("Hypothesis to test", key="loop_hyp", height=90)
        _ev_preview = ""
        try:
            from co_scientist import experiment as _exp0
            _ev_preview = _exp0.project_evidence()
        except Exception:
            pass
        if _ev_preview:
            with st.expander("What the designer will take into account", expanded=False):
                st.caption("Pulled live from your leaderboard, the cross-check "
                           "models, and any bench results on record.")
                st.code(_ev_preview)
        if not have_key:
            st.info("Add your Anthropic API key in the sidebar to design an experiment.")
        if st.button("Design an alamarBlue experiment", type="primary",
                     disabled=not have_key, key="loop_design_btn"):
            from co_scientist import experiment as _exp
            with st.spinner("Designing the cleanest test…"):
                try:
                    ev = _exp.project_evidence()
                    st.session_state.loop_evidence = ev
                    st.session_state.loop_design = _exp.design_experiment(
                        st.session_state.loop_hyp, ev)
                    _d0 = st.session_state.loop_design or {}
                    _tr("design", "Designed an alamarBlue experiment",
                        detail=str(_d0.get("aim", ""))[:200],
                        inputs={"hypothesis": st.session_state.loop_hyp[:150]},
                        outputs={"cell line": str(_d0.get("cell_line", "")),
                                 "treatment": str(_d0.get("treatment", ""))[:120],
                                 "reagents": ", ".join(
                                     str(r) for r in (_d0.get("reagents_needed") or []))[:120],
                                 "limitation": str(_d0.get("limitation", ""))[:150]})
                except Exception as ex:
                    st.error(f"Design failed: {ex}")
        d = st.session_state.get("loop_design")
        if d:
            def _txt(v):
                """Render a field whether the model returned a string, a list,
                or a nested object."""
                if isinstance(v, dict):
                    return "  \n".join(f"*{k.replace('_', ' ')}:* {_txt(x)}"
                                       for k, x in v.items())
                if isinstance(v, list):
                    return "  \n".join(f"- {_txt(x)}" for x in v)
                return str(v or "")

            st.markdown(f"**Aim.** {_txt(d.get('aim'))}")
            cc = st.columns(2)
            cc[0].markdown(f"**Cell line**\n\n{_txt(d.get('cell_line'))}")
            cc[1].markdown(f"**Treatment**\n\n{_txt(d.get('treatment'))}")
            st.markdown(f"**Key comparison.** {_txt(d.get('comparison'))}")
            if d.get("controls"):
                st.markdown("**Controls**")
                st.markdown(_txt(d["controls"]))
            st.markdown(f"**Readout.** {_txt(d.get('readout'))}")
            st.markdown(f"**Watch out for.** {_txt(d.get('key_confound'))}")
            if d.get("limitation"):
                st.warning(f"**What this can't establish.** {_txt(d['limitation'])}")
            if d.get("reagents_needed"):
                reg = d["reagents_needed"]
                reg = reg if isinstance(reg, list) else [str(reg)]
                st.success("Reagents to pull: " + ", ".join(str(r) for r in reg))
            st.caption("Next: Execute: locate these in your freezer.")

    # ---------- 2. Execute. Find the reagents ----------
    with e_tab:
        st.markdown("Check the reagents your design needs against what's in your "
                    "freezer. (Manage your boxes and lists in the **Reagent "
                    "inventory** tab.)")
        box = st.session_state.get("loop_box") or freezer.load_box(freezer.DEFAULT_BOX)
        st.caption(f"Inventory: {st.session_state.get('loop_box_name', 'demo_drug_box.csv')}")

        d = st.session_state.get("loop_design")
        _rn = (d or {}).get("reagents_needed")
        if isinstance(_rn, str):
            _rn = [_rn]
        _rn = [_clean_reagent(r) for r in (_rn or [])]
        _rn = [r for r in _rn if r]

        # Streamlit trap: a text_input with BOTH `key` and `value` only honours
        # `value` the first time. After that session_state wins, so the box kept
        # showing the demo reagents forever and the design never came through.
        # Write to session_state instead, and only when the design changes: so
        # a hand-edited list isn't clobbered on every rerun.
        _sig = "|".join(_rn)
        if _rn and st.session_state.get("_exec_design_sig") != _sig:
            st.session_state["exec_needed"] = ", ".join(_rn)
            st.session_state["_exec_design_sig"] = _sig
        st.session_state.setdefault("exec_needed", "kifunensine, bortezomib, DMSO")

        if d:
            st.caption("From your current design: "
                       + (", ".join(_rn) if _rn else "no reagents listed"))
        else:
            st.info("No design yet. Showing demo reagents. Run **1 · Design** "
                    "first and this fills itself in.")

        cn, cb = st.columns([4, 1])
        with cn:
            needed_txt = st.text_input("Reagents needed (comma-separated)",
                                       key="exec_needed")
        with cb:
            st.write("")
            st.write("")
            if st.button("Reset to design", disabled=not _rn,
                         key="exec_reset", help="Discard edits and re-pull "
                                                "the design's reagent list."):
                st.session_state["exec_needed"] = ", ".join(_rn)
                st.rerun()
        needed = [x.strip() for x in needed_txt.split(",") if x.strip()]
        st.caption("alamarBlue, media, and plates are assumed on hand. This checks "
                   "the experimental compounds.")
        if hasattr(freezer, "reconcile"):
            rec = freezer.reconcile(needed, box)
            _tr("reagents", f"Checked {len(needed)} reagent(s) against the freezer",
                outputs={r["reagent"]: (f"found at {r['position']}"
                                        if r["found"] else "not in this box")
                         for r in rec})
        else:
            # A stale deploy may hold an older freezer module without reconcile();
            # fall back to locate() so the tab still works. Reboot to refresh.
            st.caption("Running a compatibility fallback. Reboot the app to refresh.")
            rec = [{"reagent": r, "found": bool(h),
                    "position": h[0]["position"] if h else None,
                    "label": h[0]["label"] if h else None}
                   for r in needed for h in [freezer.locate(r, box)]]
        for row in rec:
            if row["found"]:
                st.markdown(f"- **{row['reagent']}**: in the box at "
                            f"**{row['position']}** ({row['label']})")
            else:
                st.markdown(f"- **{row['reagent']}**: not in this box; order it or "
                            "point Benchmate at the right box")
        missing = [r["reagent"] for r in rec if not r["found"]]
        if missing:
            st.warning("Need to source: " + ", ".join(missing))
        elif needed:
            st.success("Everything's on hand. You're ready to run.")

    # ---------- 3. Results & feedback ----------
    with r_tab:
        st.markdown("Ran the assay? Drop the rig CSV here. Benchmate scores "
                    "viability, audits it for artifacts, and files it as evidence.")
        _bench_assay_panel()

        st.divider()
        st.subheader("Feed it back. Sharpen the hypothesis")
        on_record = assay.available_assays()
        default_summary = ""
        if on_record:
            rec0 = assay.assay_evidence(on_record[-1])
            if rec0:
                default_summary = (f"{rec0['drug']} on {rec0['cell_line']}: "
                                   f"{rec0['viability']['verdict']} "
                                   f"({rec0['direction_for_benchmate']}).")
        summary = st.text_area("Bench result summary", value=default_summary,
                               height=80, key="loop_result_sum")
        if not have_key:
            st.info("Add your Anthropic API key to refine the hypothesis.")
        if st.button("Refine the hypothesis from this result",
                     disabled=not (have_key and summary.strip()), key="loop_refine_btn"):
            from co_scientist import experiment as _exp
            with st.spinner("Rethinking in light of the bench…"):
                try:
                    _tr("feedback", "Fed the bench result back to the agents")
                    st.session_state.loop_refined = _exp.refine_hypothesis(
                        st.session_state.loop_hyp, summary)
                except Exception as ex:
                    st.error(f"Refine failed: {ex}")
        ref = st.session_state.get("loop_refined")
        if ref:
            st.markdown(f"**Verdict:** {ref.get('verdict', '')}")
            st.markdown(f"**Revised hypothesis:** {ref.get('revised_hypothesis', '')}")
            st.caption(ref.get("rationale", ""))
            st.markdown(f"**Next experiment:** {ref.get('next_experiment', '')}")
            if st.button("Use the revised hypothesis in Design", key="loop_use_refined"):
                st.session_state["_pending_hyp"] = ref.get(
                    "revised_hypothesis", st.session_state.loop_hyp)
                st.rerun()

    # ---------- Reagent inventory ----------
    with inv_tab:
        st.markdown(
            "Your on-hand reagents. This is what **Execute** checks against. "
            "Upload a **CryoVision box map** (scan a photo locally: "
            "`python cryovision.py --image box.jpg --output box.csv`) or a plain "
            "**reagent list** (CSV or Excel with a name column, and optionally a "
            "location/box column)."
        )
        with st.container(border=True):
            st.markdown("**Scan a freezer photo** *(runs locally)*")
            st.caption("Needs the CryoVision repo on this machine (set `CRYOVISION_DIR` "
                       "or clone it next to Benchmate). On the hosted app, upload a "
                       "CSV/Excel map below instead.")
            img = st.file_uploader("Freezer box photo (JPG/PNG)",
                                   type=["jpg", "jpeg", "png"], key="inv_img")
            if img is not None and st.button("Parse box with CryoVision", key="inv_scan_btn"):
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    st.warning("CryoVision's vision step needs your Anthropic key. "
                               "Put it in the sidebar (and/or `export ANTHROPIC_API_KEY=...` "
                               "before launching the app), then retry.")
                elif not (hasattr(freezer, "cryovision_available") and freezer.cryovision_available()):
                    st.warning("CryoVision not found locally. Set `CRYOVISION_DIR` to the "
                               "cloned repo, or upload a CSV/Excel map below.")
                else:
                    import tempfile
                    ext = "." + img.name.rsplit(".", 1)[-1].lower()
                    tmpimg = Path(tempfile.mkdtemp()) / ("box" + ext)
                    tmpimg.write_bytes(img.getvalue())
                    with st.spinner("Reading the box with CryoVision… (up to a minute)"):
                        try:
                            inv = freezer.scan_image(tmpimg)
                            st.session_state.loop_box = inv
                            st.session_state.loop_box_name = img.name + " (scanned)"
                            st.success(f"Parsed {img.name}: "
                                       f"{sum(1 for c in inv if c['label'])} reagents read.")
                        except Exception as ex:
                            st.error(f"Scan failed: {ex}")

        fu = st.file_uploader("Inventory. Box map or reagent list (CSV, JSON, or Excel)",
                              type=["csv", "json", "xlsx", "xls"], key="inv_up")
        if fu is not None:
            import tempfile
            suffix = "." + fu.name.rsplit(".", 1)[-1].lower()
            tmp = Path(tempfile.mkdtemp()) / ("inv" + suffix)
            tmp.write_bytes(fu.getvalue())
            try:
                inv = (freezer.load_inventory(tmp)
                       if hasattr(freezer, "load_inventory") else freezer.load_box(tmp))
                st.session_state.loop_box = inv
                st.session_state.loop_box_name = fu.name
                st.success(f"Loaded {fu.name}: {len(inv)} entries, "
                           f"{sum(1 for c in inv if c['label'])} with a reagent.")
            except Exception as ex:
                st.error(f"Couldn't read that file: {ex}")

        inv = st.session_state.get("loop_box") or freezer.load_box(freezer.DEFAULT_BOX)
        st.caption(f"Active inventory: {st.session_state.get('loop_box_name', 'demo_drug_box.csv')} "
                   f"{sum(1 for c in inv if c['label'])} reagents on hand.")

        q = st.text_input("Search the inventory", key="inv_q")
        shown = [c for c in inv if c["label"] and
                 (not q or freezer._norm(q) in freezer._norm(c["label"]))]
        import pandas as pd
        if shown:
            df = pd.DataFrame([{"reagent": c["label"], "location": c["position"] or "-"}
                               for c in shown])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No matching reagents.")

    st.divider()
    benchmate_anchor("experiment", "*design an experiment for the top idea* · *where is kifunensine?* · *do I have everything?*")
