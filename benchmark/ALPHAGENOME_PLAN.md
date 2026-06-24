# Using AlphaGenome / Enformer with Benchmate — the honest plan

Wei-Hung's note: don't trust Elo alone to pick wet-lab candidates — cross-check
it against an independent quantitative predictor like AlphaGenome or Enformer.
Here's how that actually fits, including the part that's harder than it sounds.

## The mapping problem (read this first)

AlphaGenome and Enformer are **sequence-to-function** models: give them a DNA
sequence or a variant, they predict a regulatory readout (expression, splicing,
chromatin). They answer *"does this base change alter what the genome does?"*

Most of Benchmate's current hypotheses are **perturbation / mechanism** claims —
"inhibit p97," "knock down HRD1," "EDEM1 is upregulated." Those are **not**
sequence changes, so a sequence model can't score them directly. This is the
crux: you can't just point AlphaGenome at the existing gold set.

So there are three honest ways to use it, in increasing effort.

## Option A — Candidate-selection cross-check (closest to Wei-Hung's point)

For the *subset* of top hypotheses that name a regulatory variant, compute an
AlphaGenome score and correlate it with the Elo ranking
(`benchmark/elo_vs_variant_score.py`). Low correlation = a flag before anything
goes to the bench. This is a *filter on the final shortlist*, not a change to the
loop. Cheapest, but only applies to hypotheses you can express as a variant.

## Option B — A variant-centric mini gold set (the cleanest measurable experiment)

Author ~6 hypotheses that ARE about regulatory variants near ERAD genes — e.g.
"a variant in the SEL1L promoter raises SEL1L expression and confers bortezomib
resistance" — written at tiers from strong predicted effect to weak/none. Then:
1. score each variant with AlphaGenome `predict_variant` (RNA-seq output),
2. rank them by Elo (the LLM judge) and separately by AlphaGenome score,
3. report the correlation.

This gives a real, reportable number and a clean story: *where the LLM ranking
and the sequence model agree vs diverge.* It mirrors `gold_set.py` but
variant-centric. Candidate ERAD loci to build around: SEL1L, HERPUD1, EDEM1,
DERL1, XBP1, SYVN1 (promoters / known eQTLs).

## Option C — AlphaGenome as an agent tool (the "AlphaFold slot")

Wire AlphaGenome in like Geneformer: when a hypothesis proposes a regulatory
variant, the Generation/Reflection agents call it and inject the predicted effect
as evidence. This makes the *loop* quantitatively grounded, not just the
post-hoc check. Most powerful, most work; do it after A/B show value.

## Honest scope

AlphaGenome/Enformer only cover the **regulatory / expression / splicing** slice.
They say nothing about protein structure, protein-protein interactions, or
drug binding — those need AlphaFold / docking models (a different "AlphaFold
slot" tool). So "cross-check with a quantitative model" really means *a panel* of
models, each covering the hypotheses it can actually score. Be explicit in any
writeup about which hypotheses were scorable and which weren't.

## Recommended order

1. **Option B** — author the 6-variant gold set and run the Elo-vs-AlphaGenome
   correlation. It's the one that produces a clean, honest result and a figure.
2. If the correlation is informative, **Option A** as a standing shortlist check.
3. **Option C** later, once the score is trusted enough to feed the loop.

Tooling is already in place: `co_scientist/variant_scorer.py` (AlphaGenome +
Enformer backends) and `benchmark/elo_vs_variant_score.py` (the correlation
harness). The missing piece is the variant-centric hypotheses — that's the next
thing to author (with your biology eye).
