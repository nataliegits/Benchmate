"""Push a generated notebook to a GitHub Gist and return its Colab URL.

Requires the `gh` CLI to be installed and authenticated:
    brew install gh
    gh auth login

If `gh` is unavailable (e.g. when this code runs on Streamlit Cloud),
`push_to_gist` raises `GhUnavailable`. Callers should fall back to letting
the user download the notebook and upload it to Colab manually.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class GhUnavailable(RuntimeError):
    """Raised when `gh` CLI is not installed or not authenticated."""


def _find_gh() -> str | None:
    """Locate the gh binary, checking PATH plus common Homebrew install paths."""
    on_path = shutil.which("gh")
    if on_path:
        return on_path
    # Streamlit (and other GUI-launched Python processes) sometimes start with
    # a stripped PATH that doesn't include Homebrew. Check the usual spots.
    for cand in ("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/home/linuxbrew/.linuxbrew/bin/gh"):
        if Path(cand).exists():
            return cand
    return None


def gh_available() -> bool:
    """Return True if `gh` is installed and the user is authenticated."""
    gh = _find_gh()
    if gh is None:
        return False
    r = subprocess.run([gh, "auth", "status"], capture_output=True)
    return r.returncode == 0


def push_to_gist(nb_path: Path, description: str = "Benchmate Geneformer perturbation") -> str:
    """Create a gist from `nb_path` and return its raw gist URL."""
    gh = _find_gh()
    if gh is None:
        raise GhUnavailable(
            "`gh` CLI not installed or not authenticated. Install with "
            "`brew install gh && gh auth login`, or download the notebook "
            "and upload it to Colab manually."
        )
    result = subprocess.run(
        [gh, "gist", "create", str(nb_path), "--public", "--desc", description],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh gist create failed: {result.stderr}")
    # gh prints the gist URL as the last non-empty stdout line.
    url = next(line for line in result.stdout.strip().splitlines()[::-1] if line.startswith("http"))
    return url


def colab_url(gist_url: str) -> str:
    """Convert a gist URL into a Colab 'open in Colab' link.

    Colab needs the full `/gist/{username}/{gist_id}` path; gist ID alone
    won't resolve (it raises 'Unexpected GitHub Gist path').
    """
    m = re.search(r"gist\.github\.com/([^/]+)/(\w+)", gist_url)
    if not m:
        raise ValueError(f"Could not parse gist URL {gist_url}")
    username, gist_id = m.group(1), m.group(2)
    return f"https://colab.research.google.com/gist/{username}/{gist_id}"


def handoff(nb_path: Path, description: str = "Benchmate Geneformer perturbation") -> dict:
    """Push notebook to gist and return both URLs."""
    gist_url = push_to_gist(nb_path, description=description)
    return {
        "gist_url": gist_url,
        "colab_url": colab_url(gist_url),
    }


if __name__ == "__main__":
    import sys
    nb_path = Path(sys.argv[1])
    print(handoff(nb_path))
