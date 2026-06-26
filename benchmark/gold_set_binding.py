"""Binding-hypothesis gold set — the hypotheses Boltz can actually score.

AlphaGenome scores regulatory variants; Boltz scores *binding* — does a given
small molecule actually engage a given protein target. So this set reframes ERAD
/ myeloma biology as **protein-target + ligand** pairs, each with a claim about
whether the molecule inhibits the target and re-imposes proteotoxic stress.

The experiment (mirrors the AlphaGenome cross-check): rank these by the LLM judge
(Elo) AND score each pair with Boltz (binding affinity / confidence), then
correlate. Agreement is reassuring; disagreement is the flag.

Tiers (gold order, best first) = how strong the binding *should* be:
  A  a real, validated inhibitor of the named target (strong binder expected)
  B  plausible but unproven / weaker engagement
  C  a mismatched pair — the molecule shouldn't bind this target (control)

⚠️  PLACEHOLDERS. The protein sequences are abbreviated stand-ins ("MELE…") and
some pairings are illustrative. Before trusting any score: drop in the real
UniProt sequence for each target and confirm the SMILES (e.g. from PubChem/ChEMBL).
The `gene`, `drug`, and `rationale` fields are the meaningful parts; the exact
sequence/SMILES are for you to set.
"""
from __future__ import annotations

# Each entry pairs a natural-language hypothesis (what the judge ranks) with the
# protein+ligand Boltz scores. `label` ties the two together in boltz_scores.json.
GOLD_BINDING: list[dict] = [

    dict(tier="A", gene="p97/VCP", drug="CB-5083", label="p97_CB5083",
         uniprot_gene="VCP",
         protein="MELE_PLACEHOLDER_p97",          # TODO: real VCP/p97 UniProt seq
         ligand_smiles="CC(C)c1nc(N)c2cc(-c3cn(C)c4ncccc34)ccc2n1",  # CB-5083 (verify)
         statement=("CB-5083 inhibits the p97/VCP retrotranslocation motor, "
                    "trapping ubiquitinated ERAD cargo and re-imposing lethal "
                    "proteotoxic stress in bortezomib-resistant myeloma."),
         rationale=("CB-5083 is a well-characterised ATP-competitive p97 inhibitor; "
                    "strong binding to p97 is expected — a high-tier, scoreable "
                    "binding hypothesis.")),

    dict(tier="A", gene="proteasome 20S", drug="bortezomib", label="psmb5_btz",
         uniprot_gene="PSMB5",
         protein="MELE_PLACEHOLDER_PSMB5",         # TODO: real PSMB5 seq
         ligand_smiles="CC(C)C[C@@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)NB(O)O)NC(=O)c1cnccn1",  # bortezomib (verify)
         statement=("Bortezomib engages the chymotrypsin-like β5 (PSMB5) subunit "
                    "of the 20S proteasome; the binding is the basis of its "
                    "proteotoxic killing in myeloma."),
         rationale=("Bortezomib's PSMB5 binding is textbook — a positive control "
                    "for a real, strong binder.")),

    dict(tier="B", gene="HRD1 (SYVN1)", drug="LS-102", label="hrd1_ls102",
         uniprot_gene="SYVN1",
         protein="MELE_PLACEHOLDER_SYVN1",         # TODO: real SYVN1 seq
         ligand_smiles="O=C(O)c1ccccc1",           # placeholder SMILES — replace
         statement=("A small-molecule HRD1 ligase inhibitor blocks ERAD substrate "
                    "ubiquitination and resensitises resistant cells to bortezomib."),
         rationale=("HRD1 small-molecule inhibition is plausible but far less "
                    "validated than p97/proteasome — moderate confidence.")),

    dict(tier="B", gene="EDEM1", drug="kifunensine", label="edem1_kif",
         uniprot_gene="EDEM1",
         protein="MELE_PLACEHOLDER_EDEM1",         # TODO: real EDEM1 seq
         ligand_smiles="OCC1OC(O)C(O)C(O)C1O",     # placeholder sugar-like SMILES — replace
         statement=("Kifunensine inhibits the ER mannosidase EDEM1, slowing "
                    "glycoprotein triage and re-flooding the proteasome."),
         rationale=("Kifunensine is a broad mannosidase inhibitor; engagement of "
                    "EDEM1 specifically is plausible but less direct.")),

    dict(tier="C", gene="p97/VCP", drug="aspirin", label="p97_aspirin",
         uniprot_gene="VCP",
         protein="MELE_PLACEHOLDER_p97",
         ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",    # aspirin — a mismatch control
         statement=("Aspirin inhibits the p97/VCP motor and re-imposes "
                    "proteotoxic stress in resistant myeloma."),
         rationale=("Aspirin is not a p97 inhibitor — a deliberate mismatch that "
                    "should score as a weak/no binder.")),

    dict(tier="C", gene="proteasome 20S", drug="caffeine", label="psmb5_caffeine",
         uniprot_gene="PSMB5",
         protein="MELE_PLACEHOLDER_PSMB5",
         ligand_smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine — mismatch control
         statement=("Caffeine binds the 20S proteasome β5 subunit and blocks "
                    "protein degradation in myeloma."),
         rationale=("Caffeine is not a proteasome inhibitor — control pair, "
                    "expected to score low.")),
]

# gold rank = index (0 = best). Tiers run A, A, B, B, C, C.
GOLD_BINDING_RANK = {i: i for i in range(len(GOLD_BINDING))}


def _real_seqs() -> dict:
    """Load real UniProt sequences if fetch_uniprot.py has produced them."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "real_binding_seqs.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def binding_targets():
    """Build co_scientist.boltz_scorer.BoltzTarget objects for scoring.

    Uses real UniProt sequences from real_binding_seqs.json when present
    (run `python -m benchmark.fetch_uniprot`); otherwise the placeholders above.
    """
    from co_scientist.boltz_scorer import BoltzTarget
    real = _real_seqs()
    return [BoltzTarget(protein=real.get(g["label"], g["protein"]),
                        ligand_smiles=g["ligand_smiles"], label=g["label"])
            for g in GOLD_BINDING]
