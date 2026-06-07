"""A small gold-standard hypothesis set for validating the ranking.

You can't get wet-lab ground truth on demand, so we use the next best thing:
hypotheses written at deliberately different *quality tiers* for one research
goal. A trustworthy tournament should rank tier-A above tier-B above tier-C.
That gives us a cheap, repeatable answer to "are good hypotheses actually
winning?" — measured as rank correlation between the Elo result and the gold
tier order, and as judge accuracy on cross-tier pairs.

Tiers (what separates them):
  A  specific mechanism, falsifiable experiment with a control, grounded
  B  plausible but vaguer, weaker experimental design
  C  vague, circular, or effectively untestable

Edit / extend these for your own domain — keep the tier labels honest and the
within-file ordering = gold ranking (best first).
"""
from __future__ import annotations

from co_scientist.state import Hypothesis

GOLD_GOAL = ("Identify mechanisms by which relapsed AML cells resist BCL-2 "
             "inhibition (venetoclax), and propose testable combination "
             "strategies to overcome that resistance.")

# Ordered best -> worst. `tier` is the gold quality bucket.
GOLD: list[dict] = [
    dict(tier="A", statement=(
        "Venetoclax-resistant AML cells survive by shifting anti-apoptotic "
        "dependence from BCL-2 to MCL-1; co-inhibiting MCL-1 restores apoptosis."),
        rationale=("BH3 profiling shows MCL-1 priming rises after venetoclax; "
                   "MCL-1 sequesters BIM when BCL-2 is blocked."),
        experiment=("BH3-profile paired pre/post-relapse blasts; test venetoclax "
                    "+ MCL-1 inhibitor (S63845) vs each alone in PDX models; "
                    "rescue with MCL-1 overexpression as the specificity control.")),
    dict(tier="A", statement=(
        "Resistant blasts upregulate oxidative phosphorylation via increased "
        "fatty-acid oxidation; blocking CPT1a resensitises them to venetoclax."),
        rationale=("LSCs depend on OXPHOS; venetoclax survivors show elevated "
                   "OCR and FAO gene expression."),
        experiment=("Seahorse OCR on sorted resistant LSCs; venetoclax + "
                    "etomoxir (CPT1a inhibitor) colony assays; CPT1a knockdown "
                    "as genetic confirmation with a non-targeting shRNA control.")),
    dict(tier="B", statement=(
        "TP53 pathway status modulates venetoclax response in relapsed AML."),
        rationale=("TP53-mutant AML responds poorly to many therapies."),
        experiment=("Compare venetoclax sensitivity in TP53-wildtype vs "
                    "TP53-mutant patient samples.")),
    dict(tier="B", statement=(
        "Bone-marrow stromal contact protects AML cells from venetoclax."),
        rationale=("The microenvironment is known to confer drug protection."),
        experiment=("Culture blasts with and without stromal co-culture and "
                    "compare venetoclax-induced death.")),
    dict(tier="C", statement=(
        "Resistant AML cells are generally more robust and harder to kill."),
        rationale=("Relapsed disease tends to be more aggressive."),
        experiment=("Treat resistant cells with venetoclax and observe that "
                    "fewer of them die.")),
    dict(tier="C", statement=(
        "Epigenetic dysregulation contributes to venetoclax resistance in AML."),
        rationale=("Epigenetics influences gene expression broadly."),
        experiment=("Profile the epigenome of resistant cells and look for "
                    "differences.")),
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
