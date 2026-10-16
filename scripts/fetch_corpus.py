"""Fetch the detector's training text into .cache/ (gitignored).

- Positives: TeHikuMedia/nga-tautohetohe-reo, the te reo Māori spoken in the
  New Zealand Parliament as recorded in Hansard (public domain), pinned to
  one commit. Only its CSV is read; nothing from it is executed.
- Negatives: two Katherine Mansfield story collections from Project
  Gutenberg (public domain), checked against pinned SHA-256s.

    python -I scripts/fetch_corpus.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"

CORPUS_REPO = "https://github.com/TeHikuMedia/nga-tautohetohe-reo.git"
CORPUS_COMMIT = "ffb3b704816d4d4d06f0350b71e6f8bcca6014f2"

ENGLISH: dict[str, tuple[str, str]] = {
    # The Garden Party, and Other Stories
    "pg1429.txt": (
        "https://www.gutenberg.org/cache/epub/1429/pg1429.txt",
        "ca75eaf21b4448f4e59b0703eaf433e6726db4aca083ea49accf40dd024b9d79",
    ),
    # Bliss, and Other Stories
    "pg44385.txt": (
        "https://www.gutenberg.org/cache/epub/44385/pg44385.txt",
        "ca9849fca34dcc1d324d5b91d155e1fbae79c7db7f23e8f540865b571277409f",
    ),
}


def fetch_corpus() -> Path:
    dest = CACHE / "corpus"
    if not dest.exists():
        subprocess.run(["git", "clone", "--quiet", CORPUS_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", CORPUS_COMMIT], check=False)
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != CORPUS_COMMIT:
        # A shallow clone may lack the pinned commit; fetch it explicitly.
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "origin", CORPUS_COMMIT], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", CORPUS_COMMIT], check=True)
    return dest


def fetch_english() -> list[Path]:
    out_dir = CACHE / "english"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (url, sha) in ENGLISH.items():
        p = out_dir / name
        if not p.exists():
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https URL
                p.write_bytes(r.read())
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != sha:
            sys.exit(f"{name}: sha256 {digest} does not match the pinned {sha}")
        paths.append(p)
    return paths


if __name__ == "__main__":
    print(fetch_corpus())
    for p in fetch_english():
        print(p, hashlib.sha256(p.read_bytes()).hexdigest())
