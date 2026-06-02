"""Download the raw Flesh and Blood card data into data/raw/.

Source: the-fab-cube/flesh-and-blood-cards (community-maintained, open data).
We pull `card.json` (one record per UNIQUE card) plus a few small reference
files we'll likely need for filtering and validation later.

Why a script instead of a manual download:
- Reproducible: anyone (including future-you) can rebuild data/ with one command.
- The repo stays lean: the 20 MB download is git-ignored; this 1 KB script is not.

Run:  uv run python scripts/fetch_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# `develop` is the dataset's default/maintained branch.
BASE_URL = "https://raw.githubusercontent.com/the-fab-cube/flesh-and-blood-cards/develop/json/english"

# Primary data + small reference lookups. Legality lists can be added later
# once we confirm what `card.json` already embeds.
FILES = [
    "card.json",      # one record per unique card  <- primary source
    "type.json",      # card type reference
    "keyword.json",   # keyword reference
    "set.json",       # set reference
]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(filename: str, dest_dir: Path) -> None:
    url = f"{BASE_URL}/{filename}"
    dest = dest_dir / filename
    print(f"  {filename:16} <- {url}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted, fixed host)
        data = resp.read()
    dest.write_bytes(data)
    print(f"  {'':16}    {len(data):,} bytes written to {dest.relative_to(RAW_DIR.parent.parent)}")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(FILES)} files into {RAW_DIR}\n")
    for filename in FILES:
        fetch(filename, RAW_DIR)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
