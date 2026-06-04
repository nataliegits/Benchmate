"""Push a generated notebook to a GitHub Gist and return its Colab URL.

Two paths, tried in order:
  1. GitHub REST API via `GITHUB_TOKEN` env var — works anywhere with httpx.
     Set the token as a Streamlit Cloud secret to give public users
     one-click Colab links.
  2. Local `gh` CLI — works on a developer's Mac if `gh auth login` has
     been done.

If neither is available, `push_to_gist` raises `GhUnavailable`. Callers
should fall back to letting the user download the notebook and upload it
to Colab manually.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import httpx


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
    """True if we can create gists — either via GITHUB_TOKEN or local gh CLI."""
    if os.environ.get("GITHUB_TOKEN"):
        return True
    gh = _find_gh()
    if gh is None:
        return False
    r = subprocess.run([gh, "auth", "status"], capture_output=True)
    return r.returncode == 0


def _push_via_api(nb_path: Path, token: str, description: str) -> str:
    """Create a gist via the GitHub REST API. Works anywhere with `httpx`."""
    payload = {
        "description": description,
        "public": True,
        "files": {nb_path.name: {"content": nb_path.read_text()}},
    }
    r = httpx.post(
        "https://api.github.com/gists",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json=payload,
        timeout=20,
    )
    if r.status_code != 201:
        raise RuntimeError(f"GitHub API gist create failed ({r.status_code}): {r.text[:200]}")
    return r.json()["html_url"]


def push_to_gist(nb_path: Path, description: str = "Benchmate Geneformer perturbation") -> str:
    """Create a gist from `nb_path` and return its gist URL.

    Tries the GitHub API first (works on Streamlit Cloud if GITHUB_TOKEN is set),
    falls back to the local `gh` CLI.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return _push_via_api(nb_path, token, description)

    gh = _find_gh()
    if gh is None:
        raise GhUnavailable(
            "Neither GITHUB_TOKEN nor a `gh` CLI is available. "
            "Set GITHUB_TOKEN as an env var / Streamlit secret, install "
            "`gh` locally (`brew install gh && gh auth login`), or download "
            "the notebook and upload it to Colab manually."
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
