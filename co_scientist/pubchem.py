"""Look up a compound's SMILES by name, from PubChem.

Boltz needs the ligand as SMILES, and nobody has SMILES memorised. Asking a
user to go find the string for "kifunensine" and paste it correctly is the kind
of step that quietly stops an experiment from happening.

PubChem's PUG-REST API resolves a name to a structure, free and without a key.
This is a lookup, never a generation: if PubChem doesn't know the name, the
answer is None. A plausible-looking invented SMILES would silently fold a
different molecule and produce a confident, wrong binding score.

    https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""
from __future__ import annotations

from functools import lru_cache

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


@lru_cache(maxsize=512)
def smiles_for(name: str, timeout: float = 20.0) -> tuple[str | None, str | None, str]:
    """Resolve a compound name to (smiles, cid, note).

    `note` explains a failure in words a user can act on. Tries the canonical
    SMILES property first, then the isomeric form — stereochemistry matters for
    binding, so prefer isomeric when both exist.
    """
    q = (name or "").strip()
    if not q:
        return None, None, "No compound name given."

    try:
        import httpx
    except Exception:
        return None, None, "httpx isn't installed, so PubChem can't be reached."

    from urllib.parse import quote
    safe = quote(q, safe="")

    try:
        with httpx.Client(timeout=timeout,
                          headers={"Accept": "application/json"}) as client:
            # CID first, so we can report it as provenance
            r = client.get(f"{PUBCHEM}/compound/name/{safe}/cids/JSON")
            if r.status_code == 404:
                return None, None, (
                    f"PubChem has no compound called “{q}”. Check the spelling, "
                    f"or try a synonym or the brand name.")
            r.raise_for_status()
            cids = (r.json().get("IdentifierList") or {}).get("CID") or []
            if not cids:
                return None, None, f"PubChem returned no compound for “{q}”."
            cid = str(cids[0])

            # PubChem renamed these properties: IsomericSMILES -> SMILES and
            # CanonicalSMILES -> ConnectivitySMILES, with the old names
            # deprecated. Asking for an unknown property makes the whole
            # request fail, so try the current names first and fall back to the
            # legacy pair rather than betting on one naming era.
            smi = None
            for props_q in ("SMILES,ConnectivitySMILES",
                            "IsomericSMILES,CanonicalSMILES"):
                try:
                    r = client.get(f"{PUBCHEM}/compound/cid/{cid}/property/"
                                   f"{props_q}/JSON")
                    if r.status_code >= 400:
                        continue
                    props = ((r.json().get("PropertyTable") or {})
                             .get("Properties") or [{}])[0]
                    # prefer stereochemistry-bearing SMILES: binding depends on it
                    for key in ("SMILES", "IsomericSMILES",
                                "ConnectivitySMILES", "CanonicalSMILES"):
                        if props.get(key):
                            smi = props[key]
                            break
                    if smi:
                        break
                except Exception:
                    continue

            if not smi:
                # last resort: take whatever the full record calls it
                try:
                    r = client.get(f"{PUBCHEM}/compound/cid/{cid}/JSON")
                    r.raise_for_status()
                    blob = r.text
                    import re as _re
                    m = _re.search(r'"label":\s*"SMILES".*?"sval":\s*"([^"]+)"',
                                   blob, _re.S)
                    if m:
                        smi = m.group(1)
                except Exception:
                    pass

            if not smi:
                return None, cid, (f"PubChem knows CID {cid} but returned no "
                                   f"SMILES — the API's property names may have "
                                   f"changed again.")
            return smi, cid, ""
    except Exception as e:
        return None, None, f"PubChem lookup failed: {e}"


def pubchem_url(cid: str) -> str:
    return f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"


if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or ["bortezomib", "kifunensine", "CB-5083",
                               "staurosporine", "not-a-real-compound"]):
        smi, cid, note = smiles_for(n)
        print(f"\n{n}\n  cid={cid}\n  smiles={smi}\n  note={note}")
