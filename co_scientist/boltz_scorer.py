"""Boltz scoring — the structure & binding half of the quantitative panel.

AlphaGenome scores *regulatory* effects from DNA sequence. It says nothing about
whether a drug actually binds its target. Boltz (an AlphaFold3-class co-folding
model, plus the new BoltzProt-1 / BoltzMol-1 discovery pipelines) fills that gap:
given a protein and a ligand, it predicts a structure and a **binding affinity /
confidence**. So a hypothesis like "co-inhibit p97/VCP with CB-5083" — a *binding*
claim — gets an independent quantitative score, exactly like the AlphaGenome
cross-check but for structure instead of expression.

Why this is the easy one to actually run: it's a hosted **API** (no GPU, no local
weights, no Python-version headache), and the launch gives $100 of free credits.
    1. Sign up:  https://api.boltz.bio/console/signup
    2. Redeem the credits with code BOLTZLAUNCH:
       https://api.boltz.bio/console/billing/overview
    3. Create an API key in the console, then:  export BOLTZ_API_KEY=...

Everything here is fail-soft and lazy: importing this module needs no key and no
network, and any error returns None rather than raising.

⚠️ The Boltz API is brand new, so the exact endpoints/field names below follow a
standard submit→poll→result REST shape and are marked to VERIFY against the live
docs in the Boltz console. If a call errors, adjust `_submit` / `_poll` to match
what the console's API reference shows — the rest of the pipeline is agnostic.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

BOLTZ_API_URL = os.environ.get("BOLTZ_API_URL", "https://api.boltz.bio")
BOLTZ_API_KEY = os.environ.get("BOLTZ_API_KEY", "")
POLL_SECONDS = 5
POLL_MAX = 60          # ~5 min ceiling


@dataclass
class BoltzTarget:
    protein: str            # protein sequence (or a target identifier the API accepts)
    ligand_smiles: str      # the small molecule, as SMILES
    label: str = ""         # tag so a score maps back to a hypothesis


def boltz_available() -> bool:
    if not BOLTZ_API_KEY:
        return False
    try:
        import httpx  # noqa: F401
        return True
    except Exception:
        return False


def _headers() -> dict:
    return {"Authorization": f"Bearer {BOLTZ_API_KEY}",
            "Content-Type": "application/json"}


def _submit(client, t: BoltzTarget) -> str | None:
    """Submit a co-folding / affinity job; return a job id. VERIFY endpoint."""
    body = {
        "model": "boltz-2",                       # or "boltzprot-1" / "boltzmol-1"
        "protein": t.protein,
        "ligand": {"smiles": t.ligand_smiles},
        "predict_affinity": True,
    }
    r = client.post(f"{BOLTZ_API_URL}/v1/predict", json=body, headers=_headers())
    r.raise_for_status()
    out = r.json()
    return out.get("id") or out.get("job_id")


def _poll(client, job_id: str) -> dict | None:
    """Poll until the job is done; return the result payload. VERIFY endpoint."""
    for _ in range(POLL_MAX):
        r = client.get(f"{BOLTZ_API_URL}/v1/predict/{job_id}", headers=_headers())
        r.raise_for_status()
        out = r.json()
        status = str(out.get("status", "")).lower()
        if status in ("succeeded", "completed", "done"):
            return out
        if status in ("failed", "error"):
            return None
        time.sleep(POLL_SECONDS)
    return None


def _extract_score(result: dict) -> float | None:
    """Pull a single scalar 'how good is this binder' number out of the result.
    Tries common field names; larger should mean stronger/more-confident binding."""
    for key in ("affinity", "affinity_pred_value", "binding_affinity",
                "ic50", "score", "confidence", "iptm", "ptm"):
        v = result.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    # sometimes nested under "results"/"metrics"
    for outer in ("results", "metrics", "output"):
        inner = result.get(outer)
        if isinstance(inner, dict):
            got = _extract_score(inner)
            if got is not None:
                return got
    return None


def score_binding(t: BoltzTarget) -> float | None:
    """Predicted binding score for a protein+ligand pair, or None. Never raises."""
    if not boltz_available():
        return None
    try:
        import httpx
        with httpx.Client(timeout=30.0) as client:
            job = _submit(client, t)
            if not job:
                return None
            result = _poll(client, job)
            return _extract_score(result) if result else None
    except Exception as e:
        print(f"[boltz_scorer] error for {t.label or 'target'}: {e}")
        return None


if __name__ == "__main__":
    print("Boltz available:", boltz_available(),
          f"(BOLTZ_API_KEY {'set' if BOLTZ_API_KEY else 'NOT set'})")
    print("Sign up + $100 credits (code BOLTZLAUNCH): "
          "https://api.boltz.bio/console/signup")
    if boltz_available():
        demo = BoltzTarget(protein="MELE...",                 # replace with real seq
                           ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",   # aspirin, demo
                           label="demo")
        print("demo score:", score_binding(demo))
