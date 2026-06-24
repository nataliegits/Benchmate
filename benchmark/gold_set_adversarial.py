"""Adversarial gold set — the experiment Wei-Hung's feedback pointed to.

A plain "fluent-but-false" trap set isn't enough. If you tune ontology grounding
to demote anything that contradicts the knowledge graph, you risk building the
exact failure he warned about: a judge that also demotes *novel, paradigm-shifting*
hypotheses, because ontologies encode human consensus.

So this set has THREE kinds of hypothesis, and the experiment is a discrimination
test, not just "does Spearman go up":

  kind = "solid"   solid, specific, correct, well-grounded. Should rank HIGH.
  kind = "novel"   genuinely good science that is cutting-edge / under-characterised
                   and may NOT resolve cleanly in the ontology. Should ALSO rank high.
                   Grounding must NOT punish these.
  kind = "trap"    reads great — specific, falsifiable-sounding — but MISUSES a real
                   entity in a way that contradicts its canonical definition (or names
                   a fabricated entity). Should rank LOW. Grounding SHOULD catch these.

The win condition: ontology grounding pushes the TRAPS down without dragging the
NOVEL hypotheses down with them. See run_adversarial.py for the metric.

Domain: ERAD / bortezomib-resistant multiple myeloma (same as gold_set.py).

⚠️  BIOLOGY REVIEW NEEDED. These hypotheses (especially the traps and the "novel"
ones) were drafted to exercise the benchmark, not as vetted science. Sanity-check
the claims — and in particular confirm each trap's stated error is actually wrong —
before citing any result. The `why` field records the intended error for each trap.
"""
from __future__ import annotations

from co_scientist.state import Hypothesis

# Ordered best -> worst (gold ranking). `kind` drives the discrimination metric.
GOLD_ADV: list[dict] = [

    # ---------- SOLID · correct, specific, grounded (rank high) ----------
    dict(kind="solid", tier="A", statement=(
        "Bortezomib-resistant myeloma cells become disproportionately dependent on "
        "the HRD1/SEL1L ERAD complex to clear misfolded immunoglobulin light chains; "
        "co-inhibiting the p97/VCP retrotranslocation motor downstream of HRD1 "
        "restores lethal ER proteotoxicity."),
        rationale=("HRD1/SEL1L extracts misfolded ER clients to p97/VCP for proteasomal "
                   "degradation. Under proteasome inhibition, high-secretory cells lean "
                   "harder on upstream triage; blocking p97 traps ubiquitinated cargo "
                   "and drives terminal UPR."),
        experiment=("Deplete HRD1 (SYVN1) or SEL1L by siRNA in resistant vs parental "
                    "MM.1S/KMS-11 ± bortezomib; combine bortezomib + CB-5083 in PDX, "
                    "with a CB-5083-resistant p97 mutant as the specificity control.")),

    dict(kind="solid", tier="A", statement=(
        "Resistant plasma cells upregulate the ER mannosidase EDEM1 to accelerate "
        "misfolded-glycoprotein triage and offload the proteasome; selective EDEM1 "
        "inhibition re-floods the proteasome with ERAD substrates and resensitises "
        "cells to bortezomib."),
        rationale=("EDEM1 trims mannose to license misfolded glycoproteins for ERAD via "
                   "OS-9/XTP3-B handoff to HRD1. Faster trimming lowers steady-state ER "
                   "substrate burden, buffering proteasome inhibition."),
        experiment=("EDEM1 knockdown vs non-targeting shRNA in resistant lines; "
                    "pulse-chase NHK alpha-1-antitrypsin to confirm trimming drops; "
                    "score bortezomib EC50 shift; kifunensine as a pharmacological "
                    "complement; rescue with shRNA-refractory EDEM1 cDNA.")),

    # ---------- NOVEL · good science, under-characterised / not in consensus ----------
    dict(kind="novel", tier="A", statement=(
        "When HRD1 is inhibited, resistant cells compensate through the second ERAD E3 "
        "ligase gp78 (AMFR), so durable resensitisation to bortezomib requires "
        "co-inhibiting HRD1 and gp78 rather than HRD1 alone."),
        rationale=("gp78 acts downstream/parallel to HRD1 in ERAD and can ubiquitinate "
                   "overlapping substrates. Redundancy in the ligase layer is a "
                   "plausible, under-explored resistance route to single-ligase block."),
        experiment=("Single vs dual knockdown of HRD1 (SYVN1) and gp78 (AMFR) ± "
                    "bortezomib; measure substrate accumulation and viability; test "
                    "whether gp78 is upregulated specifically after HRD1 loss.")),

    dict(kind="novel", tier="B", statement=(
        "Resistant myeloma cells route excess misfolded cargo through selective "
        "ER-phagy (RETREG1/FAM134B-mediated) as a proteasome-independent escape valve; "
        "blocking ER-phagy collapses that valve and restores bortezomib sensitivity."),
        rationale=("ER-phagy can dispose of ER content independently of the proteasome. "
                   "Its contribution to proteasome-inhibitor resistance is not "
                   "established consensus, but is mechanistically coherent and testable."),
        experiment=("Knock down RETREG1/FAM134B or block autophagosome formation in "
                    "resistant lines ± bortezomib; track ER content turnover and "
                    "viability; look for ER-phagy flux upregulation under bortezomib.")),

    # ---------- TRAP · fluent, but the wrong MOLECULAR ROLE on an entity that
    # resolves — so a canonical fact contradicts the claim. These target roles a
    # text-only judge can plausibly mis-state, not famous textbook facts. ----------
    dict(kind="trap", tier="C", statement=(
        "In bortezomib-resistant cells the proteasome switches to a chaperone "
        "function, refolding misfolded immunoglobulin light chains back to native "
        "state instead of degrading them; enhancing this proteasomal refolding "
        "restores proteostasis and resensitises cells."),
        rationale=("If the proteasome refolds rather than degrades, resistant cells "
                   "could clear their light-chain burden without proteolysis, and "
                   "promoting refolding would relieve proteotoxic stress."),
        experiment=("Express a refolding-biased proteasome variant in resistant "
                    "lines and measure light-chain refolding vs degradation and "
                    "bortezomib sensitivity."),
        why=("FALSE: the proteasome is a proteolytic complex that DEGRADES "
             "ubiquitinated proteins — it is not a chaperone and does not refold "
             "substrates. The canonical proteasome term (a peptidase/degradation "
             "complex) contradicts the 'refolds' claim.")),

    dict(kind="trap", tier="C", statement=(
        "Derlin-1 is the E3 ubiquitin ligase that directly attaches ubiquitin to "
        "misfolded light chains during retrotranslocation; Derlin-1 ligase "
        "inhibitors therefore block ERAD and resensitise resistant myeloma cells."),
        rationale=("If Derlin-1 is the ligase tagging substrates for degradation, "
                   "inhibiting its catalytic activity would stall ERAD and re-impose "
                   "proteotoxic stress."),
        experiment=("Treat resistant lines with a Derlin-1 ligase-active-site "
                    "inhibitor and assay substrate ubiquitination and viability "
                    "± bortezomib."),
        why=("FALSE: Derlin-1 (DERL1) is a retrotranslocation channel component "
             "(a rhomboid pseudoprotease), NOT an E3 ubiquitin ligase — HRD1 is the "
             "ligase. The canonical 'Derlin-1 retrotranslocation complex' term "
             "contradicts the 'E3 ligase' role. (A plausible-sounding mix-up.)")),

    dict(kind="trap", tier="C", statement=(
        "HRD1 is a cytosolic serine/threonine kinase that activates the p97/VCP "
        "motor by phosphorylation; kinase-dead HRD1 mutants abolish "
        "retrotranslocation and reverse bortezomib resistance."),
        rationale=("If HRD1 kinase activity licenses p97, a kinase-dead mutant would "
                   "stall ERAD and re-sensitise cells."),
        experiment=("Express wild-type vs kinase-dead HRD1 in resistant lines; assay "
                    "p97 phosphorylation and bortezomib sensitivity."),
        why=("FALSE: HRD1 is an ER membrane E3 UBIQUITIN LIGASE, not a kinase. The "
             "canonical 'Hrd1p ubiquitin ligase complex' term contradicts the "
             "'kinase' claim. Positive control — this entity resolves cleanly.")),
]

# gold rank = index (0 = best). Solid + novel sit at the top; traps at the bottom.
GOLD_ADV_RANK = {i: i for i in range(len(GOLD_ADV))}


def adversarial_hypotheses() -> list[Hypothesis]:
    """Build Hypothesis objects in gold order, tagging .meta with kind/tier/gold."""
    hyps = []
    for i, g in enumerate(GOLD_ADV):
        h = Hypothesis.new(statement=g["statement"], rationale=g["rationale"],
                           experiment=g["experiment"])
        h.meta = {"kind": g["kind"], "tier": g["tier"], "gold": i,
                  "why": g.get("why", "")}
        hyps.append(h)
    return hyps
