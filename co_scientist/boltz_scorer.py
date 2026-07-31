"""Boltz scoring — the structure & binding half of the quantitative panel.

AlphaGenome scores *regulatory* effects from DNA. It says nothing about whether a
drug actually binds its target. Boltz (an AlphaFold3-class co-folding model) fills
that gap: given a protein and a ligand, it predicts a structure and a **binding
confidence**. So a hypothesis like "co-inhibit p97/VCP with CB-5083" — a *binding*
claim — gets an independent quantitative score, exactly like the AlphaGenome
cross-check but for structure instead of expression.

Uses the official Boltz Python SDK (`boltz_api`), not raw REST:

    pip install boltz-api          # the SDK (works on your Python 3.9)
    # sign up + redeem $100 credits (code BOLTZLAUNCH), make an API key:
    #   https://api.boltz.bio/console/signup
    export BOLTZ_API_KEY=...

We submit a protein + ligand complex with `binding: ligand_protein_binding` and
read back `output.binding_metrics.binding_confidence` (0–1; 0.7+ is the
high-confidence range) as the scalar score. Everything is fail-soft and lazy:
importing this needs no key/SDK, and any error returns None.

Docs: https://api.boltz.bio/docs/guides/predictions
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

BOLTZ_API_KEY = os.environ.get("BOLTZ_API_KEY", "")
BOLTZ_MODEL = os.environ.get("BOLTZ_MODEL", "boltz-2.1")
POLL_SECONDS = 5
POLL_MAX = 120          # ~10 min ceiling per job


@dataclass
class BoltzTarget:
    protein: str            # protein sequence (single-letter amino acids)
    ligand_smiles: str      # the small molecule, as SMILES
    label: str = ""         # tag so a score maps back to a hypothesis


def set_api_key(key: str) -> None:
    """Set the key at runtime, for a UI that collects it from the user.

    BOLTZ_API_KEY is read at import time, so setting os.environ afterwards has
    no effect on this module — which made a key pasted into Streamlit look like
    it did nothing. Updating both keeps the two in step.

    Deliberately session-only: the key stays in this process and is never
    written to disk or to a secrets file, so one person's paid key can't end up
    billing everyone who opens the app.
    """
    global BOLTZ_API_KEY
    BOLTZ_API_KEY = (key or "").strip()
    if BOLTZ_API_KEY:
        os.environ["BOLTZ_API_KEY"] = BOLTZ_API_KEY
    else:
        os.environ.pop("BOLTZ_API_KEY", None)


def sdk_installed() -> bool:
    """Whether the Boltz SDK is importable, regardless of key."""
    try:
        import boltz_api  # noqa: F401
        return True
    except Exception:
        return False


def boltz_available() -> bool:
    if not BOLTZ_API_KEY:
        return False
    return sdk_installed()


def _get(obj, *names):
    """Read a field whether the SDK returns objects or dicts."""
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def score_binding(t: BoltzTarget) -> float | None:
    """Predicted binding confidence (0–1) for a protein+ligand pair, or None.
    Higher = more confident the ligand binds. Never raises."""
    if not boltz_available():
        return None
    try:
        from boltz_api import Boltz
        client = Boltz(api_key=BOLTZ_API_KEY)
        prediction_input = {
            "entities": [
                {"type": "protein", "value": t.protein, "chain_ids": ["A"]},
                {"type": "ligand_smiles", "value": t.ligand_smiles, "chain_ids": ["B"]},
            ],
            "binding": {"type": "ligand_protein_binding", "binder_chain_id": "B"},
            "num_samples": 1,
        }
        pred = client.predictions.structure_and_binding.start(
            model=BOLTZ_MODEL, input=prediction_input)
        pid = _get(pred, "id")
        for _ in range(POLL_MAX):
            status = str(_get(pred, "status") or "").lower()
            if status in ("succeeded", "failed"):
                break
            time.sleep(POLL_SECONDS)
            pred = client.predictions.structure_and_binding.retrieve(pid)
        if str(_get(pred, "status") or "").lower() != "succeeded":
            print(f"[boltz_scorer] {t.label or 'target'}: status "
                  f"{_get(pred, 'status')} — {_get(pred, 'error')}")
            return None
        out = _get(pred, "output")
        bm = _get(out, "binding_metrics") if out is not None else None
        if bm is None:
            return None
        # binding_confidence is the headline scalar; optimization_score as backup
        v = _get(bm, "binding_confidence", "optimization_score")
        return float(v) if v is not None else None
    except Exception as e:
        print(f"[boltz_scorer] error for {t.label or 'target'}: {e}")
        return None


if __name__ == "__main__":
    print("Boltz available:", boltz_available(),
          f"(BOLTZ_API_KEY {'set' if BOLTZ_API_KEY else 'NOT set'}, model {BOLTZ_MODEL})")
    print("Setup: pip install boltz-api ; key + $100 credits (BOLTZLAUNCH) at "
          "https://api.boltz.bio/console/signup")
    if boltz_available():
        demo = BoltzTarget(protein="MKTIIALSYIFCLVFA",
                           ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                           label="demo")
        print("demo binding_confidence:", score_binding(demo))
