"""A small gold-standard hypothesis set for validating the ranking.

You can't get wet-lab ground truth on demand, so we use the next best thing:
hypotheses written at deliberately different *quality tiers* for one research
goal. A trustworthy tournament should rank tier-A above tier-B above tier-C.
That gives us a cheap, repeatable answer to "are good hypotheses actually
winning?" — measured as rank correlation between the Elo result and the gold
tier order, and as judge accuracy on cross-tier pairs.

Tiers (what separates them):
  A  specific complex / enzyme, falsifiable experiment with a control, grounded
  B  plausible but vaguer, weaker experimental design
  C  vague, circular, or effectively untestable

This set is centered on ER-associated protein degradation (ERAD): plasma cells
in multiple myeloma carry a massive immunoglobulin folding load and lean hard
on ERAD to dispose of misfolded light chains, so ERAD machinery is one of the
better-characterised vulnerabilities behind proteasome-inhibitor (bortezomib)
resistance.

Edit / extend these for your own domain — keep the tier labels honest and the
within-file ordering = gold ranking (best first).
"""
from __future__ import annotations

from co_scientist.state import Hypothesis

GOLD_GOAL = ("Identify mechanisms by which multiple-myeloma plasma cells "
             "rewire ER-associated protein degradation (ERAD) to survive "
             "proteasome inhibition (bortezomib), and propose testable "
             "combination strategies that re-impose lethal proteotoxic "
             "stress.")

# Ordered best -> worst. `tier` is the gold quality bucket.
GOLD: list[dict] = [
    dict(tier="A", statement=(
        "Bortezomib-resistant myeloma cells become disproportionately dependent "
        "on the HRD1/SEL1L ERAD complex to dispose of misfolded immunoglobulin "
        "light chains; co-inhibiting the p97/VCP retrotranslocation motor "
        "downstream of HRD1 restores lethal ER proteotoxicity."),
        rationale=("HRD1/SEL1L extracts misfolded ER clients to p97/VCP, which "
                   "unfolds them for proteasomal degradation. When the "
                   "proteasome is throttled, cells with high secretory load "
                   "depend more, not less, on upstream triage; blocking p97 "
                   "traps the ubiquitinated cargo on the ER membrane and "
                   "drives terminal UPR."),
        experiment=("In bortezomib-resistant vs parental MM.1S and KMS-11 "
                    "lines, deplete HRD1 (SYVN1) or SEL1L by siRNA and measure "
                    "viability ± bortezomib; combine bortezomib + CB-5083 "
                    "(p97 inhibitor) in PDX models and score tumour burden; "
                    "rescue with a CB-5083-resistant p97 mutant as the "
                    "specificity control.")),
    dict(tier="A", statement=(
        "Resistant plasma cells upregulate the ER mannosidase EDEM1 to "
        "accelerate misfolded-glycoprotein triage and offload the proteasome; "
        "selective EDEM1 inhibition re-floods the proteasome with ERAD "
        "substrates and resensitises cells to bortezomib."),
        rationale=("EDEM1 trims mannose residues that license misfolded "
                   "glycoproteins for ERAD via OS-9/XTP3-B handoff to HRD1. "
                   "Faster trimming means faster disposal and lower "
                   "steady-state ER substrate burden, which would buffer the "
                   "cell against proteasome inhibition."),
        experiment=("EDEM1 knockdown (vs non-targeting shRNA) in resistant "
                    "lines, then pulse-chase the model ERAD substrate "
                    "NHK alpha-1-antitrypsin to confirm trimming rate drops; "
                    "score bortezomib EC50 shift; orthogonally treat with "
                    "kifunensine (broad mannosidase inhibitor) as a "
                    "pharmacological complement and rescue with an EDEM1 "
                    "cDNA refractory to the shRNA.")),
    dict(tier="B", statement=(
        "UPR signalling status modulates bortezomib response in multiple "
        "myeloma."),
        rationale=("The unfolded protein response is known to influence "
                   "plasma-cell sensitivity to ER stress."),
        experiment=("Compare bortezomib sensitivity in XBP1-high vs XBP1-low "
                    "patient samples.")),
    dict(tier="B", statement=(
        "Bone-marrow stromal contact protects myeloma cells from bortezomib "
        "via altered ER stress."),
        rationale=("The bone-marrow microenvironment is broadly known to "
                   "confer drug protection in myeloma."),
        experiment=("Culture myeloma cells with and without HS-5 stromal "
                    "co-culture and compare bortezomib-induced death.")),
    dict(tier="C", statement=(
        "Resistant myeloma cells handle protein misfolding better and are "
        "therefore harder to kill."),
        rationale=("Relapsed disease tends to be more aggressive and "
                   "treatment-tolerant."),
        experiment=("Treat resistant cells with bortezomib and observe that "
                    "fewer of them die.")),
    dict(tier="C", statement=(
        "Proteostasis dysregulation contributes to bortezomib resistance in "
        "plasma cell malignancies."),
        rationale=("Proteostasis influences many cellular phenotypes."),
        experiment=("Profile proteostasis pathways in resistant cells and "
                    "look for differences.")),
]

# gold rank = index (0 = best). Same-tier items share a tier but keep their order.
GOLD_RANK = {i: i for i in range(len(GOLD))}


def gold_hypotheses() -> list[Hypothesis]:
    """Build Hypothesis objects (ids assigned), preserving gold order."""
    return [Hypothesis.new(statement=g["statement"], rationale=g["rationale"],
                           experiment=g["experiment"]) for g in GOLD]


def gold_pairs(skip_same_tier: bool = True) -> list[tuple[int, int, int]]:
    """All ordered index pairs (i, j) with their expected winner index.

    By default only cross-tier pairs (where the 'better' one is unambiguous)
    are returned — those are the ones a competent judge must get right.
    """
    pairs = []
    for i in range(len(GOLD)):
        for j in range(i + 1, len(GOLD)):
            if skip_same_tier and GOLD[i]["tier"] == GOLD[j]["tier"]:
                continue
            pairs.append((i, j, i))     # i < j in gold order, so i is the winner
    return pairs
