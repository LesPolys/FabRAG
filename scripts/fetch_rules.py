"""Download the FAB Comprehensive Rules into data/raw/rules/.

Two sources for the same document:

1. HTML (primary) — rules.fabtcg.com publishes the CR as one page per chapter,
   with *semantic* markup: every rule paragraph is `<p class="rule" id="cr7.0.1">`
   and every heading carries its section id (`<h2 id="cr7.1">`). Chunking can
   follow the document's real structure, and every chunk gets a citable rule
   number for free.

2. PDF (fallback / comparison) — the traditional single-file CR. Kept so we can
   later contrast clean structural chunking (HTML) with the messy
   extract-and-clean pipeline a PDF forces. NOTE: the pinned PDF is v2.10.1
   (Feb 2025) because fabtcg.com's download page is JS-rendered and blocks
   scripted fetches; the HTML site tracks the current version, so expect minor
   content skew between the two.

The keyword glossary (keyword.json) is already fetched by fetch_data.py.

Run:  uv run python scripts/fetch_rules.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CR_BASE_URL = "https://rules.fabtcg.com/en/cr"

# One page per chapter. The index page is the preface; acknowledgments are
# skipped (no rules content). Slugs come straight from the site's nav.
CHAPTERS = [
    "",  # index / preface
    "01-game-concepts",
    "02-object-properties",
    "03-zones",
    "04-game-structure",
    "05-layers-cards-abilities",
    "06-effects",
    "07-combat",
    "08-keywords",
    "09-additional-rules",
    "glossary",
]

PDF_URL = (
    "https://dhhim4ltzu1pj.cloudfront.net/media/documents/"
    "FaB_Comprehensive_Rules_v2_10_1.pdf"
)

RULES_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "rules"
HTML_DIR = RULES_DIR / "html"


def fetch(url: str, dest: Path) -> None:
    print(f"  {dest.name:32} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fabrag/0.1 (local research tool)"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted, fixed hosts)
        data = resp.read()
    dest.write_bytes(data)
    print(f"  {'':32}    {len(data):,} bytes")


def main() -> int:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(CHAPTERS)} HTML pages into {HTML_DIR}\n")
    for slug in CHAPTERS:
        url = f"{CR_BASE_URL}/{slug}/" if slug else f"{CR_BASE_URL}/"
        dest = HTML_DIR / f"{slug or 'index'}.html"
        fetch(url, dest)

    print(f"\nFetching CR PDF (v2.10.1 fallback) into {RULES_DIR}\n")
    fetch(PDF_URL, RULES_DIR / "FaB_Comprehensive_Rules_v2_10_1.pdf")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
