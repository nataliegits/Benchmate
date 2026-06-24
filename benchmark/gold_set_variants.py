"""Variant-framed gold set — the hypotheses AlphaGenome / Enformer can actually score.

The other gold sets hold *perturbation* hypotheses ("inhibit p97"), which a
sequence-to-function model can't touch. This set reframes the same ERAD biology
as *regulatory variants*: a specific base change near an ERAD gene, with a claim
about its effect on that gene's expression and on bortezomib resistance. Those
AlphaGenome CAN score.

The experiment (see ALPHAGENOME_PLAN.md, Option B): rank these by the LLM judge
(Elo) AND score each variant with AlphaGenome, then correlate. Agreement is
reassuring; disagreement is a flag that Elo alone shouldn't pick wet-lab
candidates.

Tiers (gold order, best first) = how strong the regulatory effect *should* be:
  A  variant in a well-characterised regulatory element, large expected effect
  B  plausible regulatory variant, moderate/uncertain effect
  C  variant in a region with little regulatory evidence, expected ~no effect

⚠️  COORDINATES ARE PLACEHOLDERS. The genomic positions below are illustrative
(hg38-style chr/pos/ref/alt) so the pipeline runs end-to-end — they are NOT
verified regulatory variants. Before trusting any score, replace each `variant`
with a real coordinate (e.g. a known eQTL or a promoter/enhancer SNP for that
gene) and confirm the direction of the claimed effect. The `gene` and `rationale`
are the scientifically meaningful parts; the exact base is for you to set.
"""
from __future__ import annotations

# Each entry pairs a natural-language hypothesis (what the judge ranks) with the
# variant AlphaGenome scores. label ties the two together in variant_scores.json.
GOLD_VARIANTS: list[dict] = [

    dict(tier="A", gene="SEL1L", label="SEL1L_prom",
         variant=dict(chrom="chr14", pos=81_000_000, ref="C", alt="T"),
         statement=("A promoter variant that raises SEL1L expression strengthens "
                    "the HRD1/SEL1L ERAD complex and confers bortezomib resistance "
                    "in multiple myeloma."),
         rationale=("SEL1L is the limiting adaptor of the HRD1 complex; higher "
                    "SEL1L should increase ERAD throughput and buffer proteasome "
                    "inhibition. A promoter variant that up-regulates it is a "
                    "strong, scoreable regulatory hypothesis.")),

    dict(tier="A", gene="HERPUD1", label="HERPUD1_enh",
         variant=dict(chrom="chr16", pos=56_900_000, ref="G", alt="A"),
         statement=("An enhancer variant that increases HERPUD1 expression expands "
                    "ERAD capacity and reduces sensitivity to bortezomib."),
         rationale=("HERPUD1 (HERP) is a UPR-induced ERAD component; raising it "
                    "should help resistant cells clear misfolded light chains. An "
                    "enhancer variant up-regulating it has a clear expected direction.")),

    dict(tier="B", gene="EDEM1", label="EDEM1_5utr",
         variant=dict(chrom="chr3", pos=5_230_000, ref="A", alt="G"),
         statement=("A 5'UTR variant modestly increasing EDEM1 expression "
                    "accelerates glycoprotein triage and slightly lowers bortezomib "
                    "sensitivity."),
         rationale=("EDEM1 mannosidase activity offloads the proteasome, but a UTR "
                    "variant's effect on expression is uncertain — a plausible, "
                    "moderate hypothesis.")),

    dict(tier="B", gene="DERL1", label="DERL1_intron",
         variant=dict(chrom="chr8", pos=123_900_000, ref="T", alt="C"),
         statement=("An intronic variant that mildly upregulates DERL1 improves "
                    "retrotranslocation and offers partial bortezomib protection."),
         rationale=("DERL1 forms the retrotranslocation channel; an intronic "
                    "regulatory variant could nudge expression, but the magnitude "
                    "and direction are uncertain.")),

    dict(tier="C", gene="SYVN1", label="SYVN1_intergenic",
         variant=dict(chrom="chr11", pos=64_900_000, ref="G", alt="C"),
         statement=("An intergenic variant far from SYVN1 (HRD1) raises its "
                    "expression and drives bortezomib resistance."),
         rationale=("The variant sits in a region with little regulatory evidence "
                    "for SYVN1; a large expression effect is unlikely — expected to "
                    "score low.")),

    dict(tier="C", gene="XBP1", label="XBP1_desert",
         variant=dict(chrom="chr22", pos=28_790_000, ref="A", alt="T"),
         statement=("A variant in a gene desert near XBP1 increases XBP1 output and "
                    "confers proteasome-inhibitor resistance."),
         rationale=("No known regulatory element here; the claim of a strong "
                    "expression effect is weak — a control-style low-tier variant.")),
]

# gold rank = index (0 = best). Tiers run A, A, B, B, C, C.
GOLD_VARIANTS_RANK = {i: i for i in range(len(GOLD_VARIANTS))}


def _real_coords() -> dict:
    """Load real GTEx coordinates if fetch_eqtls.py has produced them."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "real_variant_coords.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def variant_objects():
    """Build co_scientist.variant_scorer.Variant objects for scoring.

    Uses real GTEx coordinates from real_variant_coords.json when present
    (run `python -m benchmark.fetch_eqtls`); otherwise the placeholders above.
    """
    from co_scientist.variant_scorer import Variant
    real = _real_coords()
    return [Variant(label=g["label"], **(real.get(g["label"], g["variant"])))
            for g in GOLD_VARIANTS]
