"""Independent quantitative scoring of hypotheses — the "orthogonal predictor"
Wei-Hung Weng suggested checking the Elo ranking against.

The idea: for any hypothesis that names a specific regulatory *variant* or
*sequence* change, a sequence-to-function model gives a quantitative predicted
effect that is completely independent of the LLM judge. Correlate that against
the Elo ranking (see benchmark/elo_vs_variant_score.py); if they don't line up,
Elo alone isn't enough to pick wet-lab candidates.

Two backends, same interface:

  AlphaGenome (RECOMMENDED first) — Google DeepMind's model, accessed via a free
    API (no GPU, no local weights). `pip install alphagenome`, get a free
    non-commercial API key from DeepMind, set ALPHAGENOME_API_KEY. Does variant
    scoring across RNA-seq / ATAC / ChIP up to ~1 Mbp.
    Docs: https://www.alphagenomedocs.com/  ·  Repo: github.com/google-deepmind/alphagenome

  Enformer — the older, fully-open model. Runs locally via `enformer-pytorch`
    (needs torch + a GPU; Colab is the easy path, like your Geneformer notebook).

Everything is fail-soft and lazy-imported: importing this module never requires
either dependency, and any backend/network error returns None rather than raising.

⚠️  Scope: these models score *sequence/variant* claims (e.g. "variant X in the
GENE promoter lowers expression"). They do NOT score perturbation claims like
"inhibit p97" — that's not a sequence change. Apply this to the subset of
hypotheses you can frame as a regulatory variant.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Window AlphaGenome/Enformer score around a variant. AlphaGenome supports up to
# 2^20 bp; Enformer uses ~196,608 bp. Pick per backend below.
ALPHAGENOME_API_KEY = os.environ.get("ALPHAGENOME_API_KEY", "")


@dataclass
class Variant:
    chrom: str          # e.g. "chr8"
    pos: int            # 1-based position
    ref: str            # reference allele, e.g. "A"
    alt: str            # alternate allele, e.g. "G"
    label: str = ""     # free tag so you can map a score back to a hypothesis


# ---------------------------------------------------------------------------
# AlphaGenome backend (API; recommended)
# ---------------------------------------------------------------------------

def alphagenome_available() -> bool:
    if not ALPHAGENOME_API_KEY:
        return False
    try:
        import alphagenome  # noqa: F401
        return True
    except Exception:
        return False


def score_variant_alphagenome(v: Variant, *, output: str = "RNA_SEQ",
                              window: int = 131_072) -> float | None:
    """Predicted functional effect magnitude of a variant via the AlphaGenome API.

    Returns a single scalar (mean absolute predicted change in the chosen output
    track between ref and alt), or None on any error. Larger = bigger predicted
    regulatory effect.

    NOTE: this follows AlphaGenome's documented call shape, but the SDK is young
    and arg names may change — verify against the Quick Start at
    https://www.alphagenomedocs.com/ and adjust if a call errors.
    """
    if not alphagenome_available():
        return None
    try:
        import numpy as np
        from alphagenome.data import genome
        from alphagenome.models import dna_client

        model = dna_client.create(ALPHAGENOME_API_KEY)
        variant = genome.Variant(
            chromosome=v.chrom, position=v.pos,
            reference_bases=v.ref, alternate_bases=v.alt,
        )
        interval = variant.reference_interval.resize(window)
        out_type = getattr(dna_client.OutputType, output)
        pred = model.predict_variant(
            interval=interval, variant=variant,
            requested_outputs=[out_type],
            ontology_terms=[],            # all available; narrow if you like
        )
        ref_vals = np.asarray(pred.reference.get(out_type).values)
        alt_vals = np.asarray(pred.alternate.get(out_type).values)
        return float(np.mean(np.abs(alt_vals - ref_vals)))
    except Exception as e:        # fail soft — never break the harness
        print(f"[variant_scorer] AlphaGenome error for {v.label or v}: {e}")
        return None


# ---------------------------------------------------------------------------
# Enformer backend (local; Colab GPU)
# ---------------------------------------------------------------------------

def enformer_available() -> bool:
    try:
        import torch  # noqa: F401
        import enformer_pytorch  # noqa: F401
        return True
    except Exception:
        return False


def score_variant_enformer(seq_ref: str, seq_alt: str, *,
                           track_idx: int = 4980) -> float | None:
    """Predicted effect of a variant via Enformer, from two 196,608-bp sequences
    (reference and alternate, variant centered). Returns mean |Δ| on one output
    track, or None. Run this on a GPU (Colab) — it loads the full model.

    track_idx default 4980 is a CAGE/expression track in the human head; pick the
    track matching your readout from the Enformer target metadata.
    """
    if not enformer_available():
        return None
    try:
        import torch
        from enformer_pytorch import Enformer, str_to_one_hot

        model = Enformer.from_pretrained("EleutherAI/enformer-official-rough").eval()
        with torch.no_grad():
            xr = str_to_one_hot(seq_ref)[None]
            xa = str_to_one_hot(seq_alt)[None]
            pr = model(xr)["human"][0, :, track_idx]
            pa = model(xa)["human"][0, :, track_idx]
            return float((pa - pr).abs().mean())
    except Exception as e:
        print(f"[variant_scorer] Enformer error: {e}")
        return None


# ---------------------------------------------------------------------------
# Convenience: which backend is live?
# ---------------------------------------------------------------------------

def active_backend() -> str | None:
    if alphagenome_available():
        return "alphagenome"
    if enformer_available():
        return "enformer"
    return None


if __name__ == "__main__":
    print("AlphaGenome available:", alphagenome_available(),
          "| Enformer available:", enformer_available())
    print("active backend:", active_backend())
    print("\nTo use AlphaGenome: pip install alphagenome, get a free key at "
          "alphagenomedocs.com, then `export ALPHAGENOME_API_KEY=...`")
