"""Generate a parameterised copy of notebook 02 with user-specified TARGETS.

The template's TARGETS dict is rewritten with the user's gene symbols,
resolved to Ensembl IDs via mygene. Everything else in the notebook
(install cells, CELLxGENE pull, perturbation, stats) is left untouched.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Iterable

import mygene

TEMPLATE = Path(__file__).resolve().parent.parent / "notebooks" / "02_geneformer_ciliated_cells.ipynb"
OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_to_ensembl(symbols: Iterable[str]) -> dict[str, str]:
    """Look up Ensembl gene IDs for each symbol via mygene."""
    mg = mygene.MyGeneInfo()
    out: dict[str, str] = {}
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        hits = mg.query(sym, fields="symbol,ensembl.gene", species="human").get("hits", [])
        for h in hits:
            if h.get("symbol") == sym:
                ens = h.get("ensembl", {})
                eid = (ens.get("gene") if isinstance(ens, dict)
                       else ens[0]["gene"] if ens else None)
                if eid:
                    out[sym] = eid
                    break
    return out


def generate_notebook(symbols: Iterable[str],
                      out_dir: Path | None = None) -> tuple[Path, dict[str, str]]:
    """Write a parameterised notebook for the given gene symbols.

    Returns (notebook_path, resolved_ensembl_map).
    Raises ValueError if no symbols could be resolved.
    """
    targets = resolve_to_ensembl(symbols)
    if not targets:
        raise ValueError(f"Could not resolve any of {list(symbols)} to Ensembl IDs.")

    nb = json.loads(TEMPLATE.read_text())

    # Build the replacement TARGETS source — one dict literal, indented for readability.
    target_lines = ["TARGETS = {\n"]
    for sym, eid in targets.items():
        target_lines.append(f'    "{sym}":  "{eid}",\n')
    target_lines.append("}\n")

    # Find the cell that defines TARGETS and replace its source.
    found = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if re.search(r"TARGETS\s*=\s*\{", src):
            # Keep any imports / preamble before the TARGETS literal
            pre_match = re.search(r"^(.*?)(TARGETS\s*=\s*\{.*?\n\})", src, re.DOTALL)
            preamble = pre_match.group(1) if pre_match else ""
            after_match = re.search(r"TARGETS\s*=\s*\{.*?\n\}(.*)$", src, re.DOTALL)
            tail = after_match.group(1) if after_match else ""
            cell["source"] = (
                [preamble] if preamble else []
            ) + target_lines + ([tail] if tail else [])
            found = True
            break
    if not found:
        raise RuntimeError("Template lacks a TARGETS = {...} cell to replace.")

    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(targets) if len(targets) <= 4 else f"{len(targets)}genes"
    out_path = out_dir / f"perturb_{suffix}_{uuid.uuid4().hex[:6]}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path, targets


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["TXNDC15", "SYVN1", "MARCHF6"]
    nb_path, resolved = generate_notebook(syms)
    print("resolved:", resolved)
    print("wrote:", nb_path)
