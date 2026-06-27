"""Gene-level gold set — hypotheses Open Targets and DepMap can score.

Open Targets scores gene↔disease *association*; DepMap scores gene *dependency*.
Both are gene-level, so this set is one hypothesis per gene: "gene X is a key
dependency/driver in bortezomib-resistant multiple myeloma." We rank these with
the LLM judge (Elo) and cross-check against each external score.

Tiers (gold order, best first) = expected disease relevance:
  A  central to the ERAD / proteasome axis in myeloma
  B  plausible / supporting
  C  weak — including a deliberate negative control (an olfactory receptor that
     has nothing to do with myeloma) to check the external models score it low.

⚠️  Tiers are informed guesses, not ground truth — that's the whole point of
cross-checking them against independent data. Disease is multiple myeloma
(Open Targets EFO_0001378).
"""
from __future__ import annotations

DISEASE = "multiple myeloma"   # resolved to an Open Targets id by name at query time

GOLD_GENES: list[dict] = [
    dict(tier="A", symbol="SYVN1", label="SYVN1",
         statement=("SYVN1 (HRD1), the core ERAD E3 ligase, is a key dependency in "
                    "bortezomib-resistant multiple myeloma."),
         rationale="HRD1/SEL1L clears misfolded light chains; resistant cells lean on it."),
    dict(tier="A", symbol="PSMB5", label="PSMB5",
         statement=("PSMB5, the chymotrypsin-like proteasome subunit bortezomib "
                    "targets, is central to proteasome-inhibitor biology in myeloma."),
         rationale="PSMB5 is the direct drug target; mutations drive resistance."),
    dict(tier="B", symbol="SEL1L", label="SEL1L",
         statement=("SEL1L, the HRD1 adaptor, supports ERAD throughput in resistant "
                    "myeloma cells."),
         rationale="Limiting adaptor of the HRD1 complex."),
    dict(tier="B", symbol="EDEM1", label="EDEM1",
         statement=("EDEM1 mannosidase accelerates ERAD glycoprotein triage and may "
                    "buffer proteasome inhibition in myeloma."),
         rationale="Speeds misfolded-glycoprotein disposal."),
    dict(tier="C", symbol="XBP1", label="XBP1",
         statement=("XBP1, the UPR transcription factor, modulates plasma-cell "
                    "sensitivity to bortezomib."),
         rationale="UPR status influences ER-stress response; indirect."),
    dict(tier="C", symbol="OR2T1", label="OR2T1_control",
         statement=("OR2T1, an olfactory receptor, is a key dependency in "
                    "bortezomib-resistant multiple myeloma."),
         rationale="Negative control — no plausible link to myeloma; should score low."),
]

GOLD_GENES_RANK = {i: i for i in range(len(GOLD_GENES))}
