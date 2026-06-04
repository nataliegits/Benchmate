"""Push a generated notebook to a GitHub Gist and return its Colab URL.

Requires the `gh` CLI to be installed and authenticated:
    brew install gh
    gh auth login

The Gist is created public-but-unlisted; only people with the URL can find it.
If you'd rather keep notebooks private, change `--public` to `--secret` below
(Colab still works with secret gists when the URL is known).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def push_to_gist(nb_path: Path, description: str = "Benchmate Geneformer perturbation") -> str:
    """Create a gist from `nb_path` and return its raw gist URL."""
    result = subprocess.run(
        ["gh", "gist", "create", str(nb_path), "--public", "--desc", description],
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
