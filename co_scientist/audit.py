"""Audit — the "is this a result or an artifact?" guard (Benchpress-style).

Before a bench result is allowed to move a hypothesis's rank, we sanity-check the
raw colour trace for physically implausible readings. The reproducibility problem
usually isn't fraud — it's a bubble, a saturated well, or a blanking error that
looks like a result. This catches the common ones so an artifact can't silently
down-weight a good hypothesis.

Checks on an alamarBlue run (rows of {t_s, R, G, B, red_blue}):
  - saturated : a colour channel pinned near the 16-bit ceiling (clipped)
  - dark      : all channels near zero (sensor dark / blanking / LED off)
  - discontinuity : red/blue jumps within a single step (bubble, bump, refocus)
  - short_run : too few reads to trust a plateau

`severe` flags (saturated / dark / short_run) mean the run should NOT feed back
into the ranking until re-run. A discontinuity is a warning, not a block.
"""
from __future__ import annotations

import math

SAT_CEILING = 60000.0   # near the 16-bit sensor ceiling -> clipped
DARK_FLOOR = 20.0        # sum of channels this low -> sensor dark / blank
JUMP_DELTA = 0.08        # red/blue change within one step -> disturbance
MIN_POINTS = 10


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def audit_run(rows: list[dict]) -> dict:
    """Return {ok, severe, flags[]} for a rig run."""
    flags: list[str] = []
    if len(rows) < MIN_POINTS:
        flags.append(f"short_run: only {len(rows)} reads — too short to trust a plateau")

    chans = [(r.get("R"), r.get("G"), r.get("B")) for r in rows]
    valid = [(R, G, B) for R, G, B in chans if all(_finite(v) for v in (R, G, B))]
    if valid:
        maxch = max(max(t) for t in valid)
        minsum = min(sum(t) for t in valid)
        if maxch >= SAT_CEILING:
            flags.append(f"saturated: a colour channel hit {maxch:.0f} (near sensor ceiling) — clipped")
        if minsum <= DARK_FLOOR:
            flags.append("dark: channels near zero — sensor dark or blanking error (check LED / wiring)")

    # A real alamarBlue trace rises monotonically to a plateau. The artifact
    # signatures are a DROP (bubble passing the sensor) or a SPIKE that falls
    # straight back (a bump / refocus) — not the fast, legitimate initial rise.
    seq = [(r["t_s"], r["red_blue"]) for r in rows if _finite(r.get("red_blue"))]
    for i in range(1, len(seq)):
        d = seq[i][1] - seq[i - 1][1]
        if d <= -JUMP_DELTA:
            flags.append(f"discontinuity: red/blue dropped {d:+.2f} at t={seq[i][0]:.0f}s "
                         "— possible bubble/disturbance")
            break
        if (d >= JUMP_DELTA and i + 1 < len(seq)
                and (seq[i + 1][1] - seq[i][1]) <= -JUMP_DELTA):
            flags.append(f"discontinuity: red/blue spiked at t={seq[i][0]:.0f}s then fell back "
                         "— possible bump")
            break

    severe = any(f.startswith(("saturated", "dark", "short_run")) for f in flags)
    return {"ok": not severe, "severe": severe, "flags": flags}


def audit_summary(a: dict) -> str:
    if not a["flags"]:
        return "Audit: clean — no artifacts detected."
    head = "Audit FLAGGED (do not trust until re-run)" if a["severe"] else "Audit: warnings"
    return head + ":\n" + "\n".join(f"  - {f}" for f in a["flags"])
