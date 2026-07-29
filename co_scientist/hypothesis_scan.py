"""Pull the scoreable objects out of a free-text hypothesis.

The cross-check models don't take sentences. Open Targets wants a gene symbol,
DepMap wants a gene symbol, AlphaMissense wants (chrom, pos, ref, alt). So
before anything can be scored, something has to read a hypothesis like

    "Elevated SEL1L sustains ERAD throughput, blunting bortezomib stress"

and come back with: genes = [SEL1L]. Not ERAD (a pathway), not bortezomib (a
drug), not the capitalised first word of the sentence.

Two rules shape the implementation:

1. Never invent. A symbol only counts as a gene if an authoritative source
   (mygene, backed by NCBI/Ensembl) confirms it. A regex alone would happily
   hand back ERAD, DMSO, UPR and CRISPR.
2. Never fabricate coordinates. AlphaMissense needs a real genomic position;
   if the hypothesis doesn't name a variant, the honest answer is "not
   applicable", not a guessed locus.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Uppercase-ish tokens that look like they could be gene symbols.
# Deliberately permissive — validation is what decides.
_CANDIDATE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{1,4})?\b")

# Things that match the shape of a gene symbol but aren't one. Checking these
# locally saves a network round-trip and, more importantly, protects against a
# validator that returns a spurious hit (several of these DO resolve to
# something in gene databases — "ERAD" and "UPR" have entries as pathway
# aliases in places).
_STOPLIST = {
    # pathways / processes / concepts
    "ERAD", "UPR", "UPS", "ER", "ERES", "ERGIC", "ERQC", "QC", "DNA", "RNA",
    "MRNA", "CRNA", "SIRNA", "SHRNA", "GRNA", "CDNA", "ATP", "ADP", "GTP",
    "NADH", "ROS", "PTM", "KO", "WT", "IC50", "EC50", "GO", "KEGG",
    # techniques / reagents / units
    "CRISPR", "CAS9", "PCR", "QPCR", "ELISA", "FACS", "WB", "IP", "CO",
    "DMSO", "PBS", "FBS", "EDTA", "TRIS", "SDS", "PAGE", "HPLC", "MS",
    "NM", "UM", "MM", "ML", "UL", "MG", "KG", "H", "HR", "HRS", "MIN",
    # study / disease / general English that survives the regex
    "MM", "AML", "CLL", "ALL", "AND", "OR", "NOT", "THE", "A", "AN", "IF",
    "IN", "ON", "OF", "TO", "BY", "AT", "IS", "IT", "WE", "US", "I",
    "NEW", "OLD", "HIGH", "LOW", "TOP", "KEY", "ONE", "TWO", "USA", "US",
    # model systems / cell lines (real, but not genes)
    "HELA", "HEK", "HEK293", "U2OS", "K562", "MM1S", "RPMI", "OPM2", "JJN3",
    "AMO1", "L363", "KMS11", "NCIH929", "H929",
    # drugs commonly named in these hypotheses
    "DMSO", "MG132", "CB", "PS341", "BTZ", "CFZ", "IXA",
}

# p.Arg175His / p.R175H style protein change
_PROT = re.compile(r"\bp\.\(?([A-Z][a-z]{2}|[A-Z])(\d+)([A-Z][a-z]{2}|[A-Z])\)?")
# c.524G>A style coding change
_CDNA = re.compile(r"\bc\.(\d+)([ACGT])>([ACGT])\b")
# rs12345
_RSID = re.compile(r"\brs\d{3,}\b")
# chr17:7676154 G>A  /  17:7676154:G:A
_GENOMIC = re.compile(
    r"\b(?:chr)?([1-9]|1[0-9]|2[0-2]|X|Y|MT)[:\s]\s?(\d{3,})[:\s]\s?"
    r"([ACGT]+)[>:\s]\s?([ACGT]+)\b")


@lru_cache(maxsize=2048)
def is_gene(symbol: str) -> bool | None:
    """Is `symbol` a real human gene symbol?

    Returns True / False / **None**, where None means "couldn't check" — the
    validator was unreachable. That third state matters: failing closed would
    turn an offline moment into the message "no gene symbol found", which reads
    as a bad hypothesis rather than a network problem. Callers should fall back
    to the regex + stoplist and say the check was skipped.

    Tries mygene (NCBI/Ensembl-backed) first, then Open Targets, which the
    cross-check panel already depends on and needs no API key.

    Cached — the same handful of symbols gets re-checked on every rerun.
    """
    sym = symbol.strip().upper()
    if not sym:
        return False
    if sym in _STOPLIST:
        return False

    try:
        import mygene
        hits = mygene.MyGeneInfo().query(
            sym, fields="symbol", species="human", size=5).get("hits", [])
        # require an EXACT match — mygene fuzzily returns neighbours
        return any(str(h.get("symbol", "")).upper() == sym for h in hits)
    except Exception:
        pass

    try:
        from .target_scorer import _resolve
        return bool(_resolve(sym, "target"))
    except Exception:
        return None


def genes_in(text: str, validate: bool = True) -> tuple[list[str], bool]:
    """Gene symbols in `text`, in order of first appearance.

    Returns (symbols, validated) where validated=False means every symbol got
    through on the regex + stoplist alone because no validator was reachable.

    Set validate=False to skip the network check deliberately.
    """
    seen: list[str] = []
    could_validate = validate
    for m in _CANDIDATE.finditer(text or ""):
        sym = m.group(0).upper()
        if sym in _STOPLIST or sym in seen or len(sym) < 2:
            continue
        if validate:
            ok = is_gene(sym)
            if ok is None:
                could_validate = False   # unreachable — keep it, flag the run
            elif not ok:
                continue
        seen.append(sym)
    return seen, could_validate


def variants_in(text: str) -> list[dict]:
    """Variant references found in `text`.

    Each entry has a `kind` and whatever fields that notation actually carries.
    Only 'genomic' entries are directly scoreable by AlphaMissense; the others
    are recorded so the UI can say "you named a variant but not its
    coordinates" rather than silently ignoring it.
    """
    out: list[dict] = []
    for m in _GENOMIC.finditer(text or ""):
        out.append({"kind": "genomic", "chrom": m.group(1),
                    "pos": int(m.group(2)), "ref": m.group(3).upper(),
                    "alt": m.group(4).upper(), "raw": m.group(0)})
    for m in _PROT.finditer(text or ""):
        out.append({"kind": "protein", "raw": m.group(0)})
    for m in _CDNA.finditer(text or ""):
        out.append({"kind": "cdna", "raw": m.group(0)})
    for m in _RSID.finditer(text or ""):
        out.append({"kind": "rsid", "raw": m.group(0)})
    return out


def scan(text: str, validate: bool = True) -> dict:
    """Everything scoreable in a hypothesis.

    Returns {genes, variants, scoreable_variants, notes} where `notes` explains
    in plain language why a model may not apply — that explanation is the whole
    point, since "no score" is otherwise indistinguishable from "broken".
    """
    genes, validated = genes_in(text, validate=validate)
    variants = variants_in(text)
    scoreable = [v for v in variants if v["kind"] == "genomic"]
    notes: list[str] = []
    if validate and not validated:
        notes.append("Couldn't reach a gene-symbol validator (mygene / Open "
                     "Targets), so these symbols were matched by pattern only "
                     "— one of them may not be a real gene.")
    if not genes:
        notes.append("No gene symbol found, so Open Targets and DepMap have "
                     "nothing to look up. Name the gene explicitly (e.g. SEL1L "
                     "rather than 'the ERAD receptor').")
    if not variants:
        notes.append("No variant named, so AlphaMissense and AlphaGenome don't "
                     "apply — both score a specific change at a specific "
                     "position.")
    elif not scoreable:
        kinds = ", ".join(sorted({v["kind"] for v in variants}))
        notes.append(f"Found a variant reference ({kinds}) but no genomic "
                     f"coordinates. AlphaMissense needs chrom/pos/ref/alt — "
                     f"look it up in ClinVar or dbSNP rather than guessing.")
    return {"genes": genes, "validated": validated, "variants": variants,
            "scoreable_variants": scoreable, "notes": notes}


if __name__ == "__main__":
    import sys
    for t in (sys.argv[1:] or [
            "Elevated SEL1L sustains ERAD throughput, blunting bortezomib stress",
            "p97/VCP inhibition with CB-5083 re-imposes lethal proteotoxic stress",
            "The TP53 variant 17:7676154 G>A disrupts DNA binding",
            "Knocking down DERL1 and SYVN1 sensitises MM1S cells to DMSO vehicle"]):
        r = scan(t, validate=False)
        print(f"\n{t}\n  genes={r['genes']}\n  variants={[v['raw'] for v in r['variants']]}")
