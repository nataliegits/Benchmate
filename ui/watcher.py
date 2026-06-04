"""Watch the local Google Drive sync folder for new perturbation CSVs and copy
them into `data/geneformer/` so the Co-Scientist agents pick them up.

Requires Google Drive desktop app installed and syncing the
`benchmate_geneformer_cilia` folder locally. The default Drive root on macOS
is `~/Library/CloudStorage/GoogleDrive-*/My Drive/...` but older installs
mount at `~/Google Drive/My Drive/...` — both are tried below.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CACHE = REPO_ROOT / "data" / "geneformer"


def detect_drive_perturb_dir() -> Path | None:
    """Find the local Google Drive folder containing notebook 02's outputs."""
    candidates = [
        Path.home() / "Library" / "CloudStorage",
        Path.home() / "Google Drive",
    ]
    needle = Path("benchmate_geneformer_cilia") / "perturbations"
    for base in candidates:
        if not base.exists():
            continue
        # CloudStorage has subdirs per Google account
        for account_dir in (base.iterdir() if base.name == "CloudStorage" else [base]):
            mydrive = account_dir / "My Drive"
            target = mydrive / needle if mydrive.exists() else account_dir / needle
            if target.exists():
                return target
    return None


class StatsCopier(FileSystemEventHandler):
    """Copy any new `*_stats.csv` into the local cache."""

    def __init__(self, on_new_csv: Callable[[Path], None] | None = None):
        self.on_new_csv = on_new_csv or (lambda p: None)

    def _maybe_copy(self, path_str: str) -> None:
        if not path_str.endswith("_stats.csv"):
            return
        src = Path(path_str)
        if not src.exists():
            return
        LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
        dst = LOCAL_CACHE / src.name
        shutil.copy(src, dst)
        print(f"[watcher] cached {dst.name}")
        self.on_new_csv(dst)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_copy(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_copy(event.src_path)


def watch(drive_dir: Path | None = None,
          on_new_csv: Callable[[Path], None] | None = None) -> Observer:
    """Start watching `drive_dir`. Returns the Observer so caller can stop it."""
    drive_dir = drive_dir or detect_drive_perturb_dir()
    if drive_dir is None:
        raise RuntimeError(
            "Could not find the Google Drive sync folder for "
            "`benchmate_geneformer_cilia/perturbations`. Make sure Google "
            "Drive desktop is installed and the folder is synced."
        )
    obs = Observer()
    obs.schedule(StatsCopier(on_new_csv=on_new_csv), str(drive_dir), recursive=True)
    obs.start()
    print(f"[watcher] watching {drive_dir}")
    print(f"[watcher] copying *_stats.csv into {LOCAL_CACHE}")
    return obs


if __name__ == "__main__":
    obs = watch()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
        obs.join()
        print("[watcher] stopped")
