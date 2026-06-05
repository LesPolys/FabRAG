"""The rules-text corpus — Phase 4, where CHUNKING enters the picture.

Cards gave us a free pass: each card is naturally one document. Prose is the
general case — a 400 KB rulebook is far too big to embed whole (and a single
vector for the whole book would be uselessly vague), so we must split it into
chunks. Every choice here is a retrieval-quality knob:

  - TOO SMALL  -> chunks lack context ("the attack" ... which attack?), and
                  related sentences land in different vectors.
  - TOO BIG    -> the embedding averages many topics into mush, and we waste
                  the LLM's context window on irrelevant neighbors.
  - ARBITRARY  -> fixed-size windows cut sentences and rules in half; the
                  classic fix is overlapping windows (duplicate N chars across
                  the cut) to make it *likely* no idea is severed.

We can do better than arbitrary, because rules.fabtcg.com publishes the
Comprehensive Rules as *semantic* HTML: every rule paragraph carries its rule
number (`<p class="rule" id="cr7.0.3">`), subrules are marked, and headings
carry section ids. So we chunk along the document's REAL structure:

  rule group  = a rule + its subrules + attached examples/notes  (atomic)
  chunk       = consecutive rule groups from ONE section, packed up to a
                size target; an oversized group splits at subrule boundaries,
                with the parent rule's text repeated in each part (structural
                "overlap": context is duplicated only where we actually cut).

Each chunk keeps its rule ids, so generated answers can cite "CR 7.0.3" and
deep-link to the official page — grounding you can verify.

Three sources flow into one corpus of RuleChunks:
  - the CR chapters (kind="rule")
  - the CR glossary (kind="glossary", one term per chunk — see note below)
  - keyword.json from the card dataset (kind="keyword", concise reminders)
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .cards import expand_symbols

# Resolved relative to this file (src/fabrag/rules.py -> repo root).
_DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
RULES_HTML_DIR = _DATA_RAW / "rules" / "html"
RAW_KEYWORD_FILE = _DATA_RAW / "keyword.json"

CR_BASE_URL = "https://rules.fabtcg.com/en/cr"

# Chapter pages that contain actual rules. index.html (the preface) is
# deliberately excluded: it's meta-text about the document itself, which would
# only ever be retrieval noise for game questions.
CHAPTER_SLUGS = [
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

# Size knobs (characters ~ 4 chars/token, so 1500 chars ~ 375 tokens).
# These are *tuning parameters* — Phase 5's eval harness will let us measure
# whether other values retrieve better instead of guessing.
DEFAULT_TARGET_CHARS = 1500   # stop packing a chunk once it passes this
DEFAULT_MAX_CHARS = 3000      # a single rule group beyond this gets split


# =============================================================================
# The chunk model
# =============================================================================
class RuleChunk(BaseModel):
    """One retrievable piece of rules text, with enough metadata to cite it."""

    chunk_id: str                                  # "cr7.0.3" (+ "/p2" if a split part)
    kind: Literal["rule", "glossary", "keyword"]
    chapter: str                                   # "7 Combat"
    section: str                                   # "7.2 Attack Step" (== chapter if none)
    rule_ids: list[str]                            # every rule/subrule id inside
    title: str | None = None                       # glossary term / keyword name
    text: str
    source_url: str | None = None                  # deep link to the first rule's anchor

    @property
    def doc_id(self) -> str:
        """Stable id under the retrieval Document protocol (see retrieval.py)."""
        return self.chunk_id

    @property
    def citation(self) -> str:
        """How an answer should refer to this chunk, e.g. 'CR 7.0.3'."""
        if self.kind == "keyword":
            return f"Keyword: {self.title}"
        if self.kind == "glossary":
            return f"CR Glossary: {self.title}"
        first = self.rule_ids[0].removeprefix("cr") if self.rule_ids else "?"
        return f"CR {first}"

    @property
    def text_for_embedding(self) -> str:
        """The text we embed: the chunk PLUS a breadcrumb of where it lives.

        A chunk read in isolation loses its surroundings — "7.2.1 The attack
        step begins..." embeds better when the vector also knows it's about
        Combat. Prepending document context to each chunk is a cheap, standard
        trick (cf. "contextual retrieval") that costs a few tokens and
        measurably helps prose corpora. Cards didn't need it: a card is its
        own complete context.
        """
        if self.kind == "keyword":
            return f"FAB keyword glossary\n{self.text}"
        return f"FAB Comprehensive Rules > {self.chapter} > {self.section}\n{self.text}"


# =============================================================================
# HTML -> blocks: a small stdlib parser for a known, regular document
# =============================================================================
# We use html.parser instead of BeautifulSoup deliberately: the markup is
# machine-generated and perfectly regular, the parse is ~60 lines, and writing
# it teaches what tree-walking libraries actually do under the hood.

_HEADINGS = {"h1", "h2"}


class _Block:
    """A flat unit of parsed content: a heading, a rule, or a subrule.

    Examples/notes/plain definition paragraphs don't become blocks of their
    own — they append to the block they follow, because that's what they
    describe (a blockquote example illustrates the rule above it; a glossary
    definition defines the term above it).
    """

    def __init__(self, kind: str, id_: str) -> None:
        self.kind = kind          # "h1" | "h2" | "rule" | "subrule"
        self.id = id_             # e.g. "cr7.0.3" ("" for id-less content)
        self.parts: list[str] = []  # text fragments, joined/normalized later

    @property
    def text(self) -> str:
        # Collapse runs of whitespace; <p> boundaries inserted "\n" markers.
        joined = "".join(self.parts)
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in joined.split("\n")]
        return "\n".join(ln for ln in lines if ln)


class _ChapterParser(HTMLParser):
    """Walk one chapter page and emit _Blocks for the article content only.

    State machine notes:
      - Everything outside <article> is site chrome (nav sidebars full of
        links) — ignored via the `in_article` flag.
      - <a class="headerlink"> is the "§" permalink decoration on every
        heading/rule — its text is suppressed.
      - <img> tags are inline game symbols; we keep their alt text ("{p}"),
        which expand_symbols() later turns into words ("power").
      - <p> with no class, <blockquote>, and <li> all *attach* to the current
        block rather than starting one.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[_Block] = []
        self.in_article = False
        self._article_depth = 0     # nesting count so we know when article ends
        self._suppress_depth = 0    # >0 while inside a headerlink anchor
        self._current: _Block | None = None

    # -- tag boundaries -------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        cls = a.get("class") or ""

        if tag == "article":
            self.in_article = True
            self._article_depth = 1
            return
        if not self.in_article:
            return
        if tag in ("article", "div", "section"):
            self._article_depth += 1

        if tag == "a" and "headerlink" in cls:
            self._suppress_depth += 1
        elif tag in _HEADINGS:
            self._start_block(tag, a.get("id") or "")
        elif tag == "p" and cls in ("rule", "subrule"):
            self._start_block(cls, a.get("id") or "")
        elif tag == "p" and self._current is not None:
            self._current.parts.append("\n")    # plain <p>: new line, same block
        elif tag == "li" and self._current is not None:
            self._current.parts.append("\n- ")  # list item: bullet line, same block
        elif tag == "img":
            alt = a.get("alt") or ""
            if self._current is not None and alt:
                self._current.parts.append(alt)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_article:
            return
        if tag in ("article", "div", "section"):
            self._article_depth -= 1
            if self._article_depth <= 0:
                self.in_article = False
                self._current = None
        elif tag == "a" and self._suppress_depth:
            self._suppress_depth -= 1
        elif tag in _HEADINGS:
            self._current = None  # heading text ends at its close tag

    def handle_data(self, data: str) -> None:
        if self.in_article and self._current is not None and not self._suppress_depth:
            self._current.parts.append(data)

    def _start_block(self, kind: str, id_: str) -> None:
        block = _Block(kind, id_)
        self.blocks.append(block)
        self._current = block


def parse_chapter(html: str) -> list[_Block]:
    parser = _ChapterParser()
    parser.feed(html)
    return [b for b in parser.blocks if b.text]


# =============================================================================
# Blocks -> rule groups -> chunks
# =============================================================================
class _RuleGroup:
    """The atomic chunking unit: one rule + its subrules (incl. attachments)."""

    def __init__(self, rule: _Block) -> None:
        self.rule = rule
        self.subrules: list[_Block] = []

    @property
    def ids(self) -> list[str]:
        return [b.id for b in (self.rule, *self.subrules) if b.id]

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in (self.rule, *self.subrules))

    def __len__(self) -> int:
        return len(self.text)


def _chunks_from_section(
    groups: list[_RuleGroup],
    *,
    kind: Literal["rule", "glossary"],
    chapter: str,
    section: str,
    slug: str,
    target_chars: int,
    max_chars: int,
) -> list[RuleChunk]:
    """Pack one section's rule groups into chunks.

    Greedy packing: keep appending whole groups until the chunk passes
    `target_chars`. A chunk never crosses a section boundary (the caller feeds
    us one section at a time) — a size-9 "Combat" tail glued onto a "Zones"
    intro would embed as mush.
    """
    chunks: list[RuleChunk] = []

    def emit(batch: list[_RuleGroup]) -> None:
        first_id = batch[0].ids[0] if batch[0].ids else ""
        chunks.append(RuleChunk(
            chunk_id=first_id or f"{slug}-{len(chunks)}",
            kind=kind,
            chapter=chapter,
            section=section,
            rule_ids=[i for g in batch for i in g.ids],
            title=_glossary_term(batch[0].rule.text) if kind == "glossary" else None,
            text=expand_symbols("\n".join(g.text for g in batch)),
            source_url=f"{CR_BASE_URL}/{slug}/#{first_id}" if first_id else None,
        ))

    def emit_split(group: _RuleGroup) -> None:
        """An oversized group splits at subrule boundaries; every part repeats
        the parent rule's text. That repetition is our 'overlap' — applied
        surgically at the one place we were forced to cut, instead of blindly
        at every window edge."""
        parent = group.rule
        part: list[_Block] = []
        parts: list[list[_Block]] = []
        size = len(parent.text)
        for sub in group.subrules:
            if part and size + len(sub.text) > target_chars:
                parts.append(part)
                part, size = [], len(parent.text)
            part.append(sub)
            size += len(sub.text)
        if part:
            parts.append(part)
        for n, subs in enumerate(parts, start=1):
            chunks.append(RuleChunk(
                chunk_id=f"{parent.id}/p{n}",
                kind=kind,
                chapter=chapter,
                section=section,
                rule_ids=[b.id for b in (parent, *subs) if b.id],
                text=expand_symbols("\n".join(b.text for b in (parent, *subs))),
                source_url=f"{CR_BASE_URL}/{slug}/#{parent.id}",
            ))

    batch: list[_RuleGroup] = []
    batch_size = 0
    for group in groups:
        if len(group) > max_chars and group.subrules:
            if batch:
                emit(batch)
                batch, batch_size = [], 0
            emit_split(group)
            continue
        # Glossary terms are independent of their alphabetical neighbors, so
        # packing them would create incoherent multi-topic chunks; rules in a
        # section flow together, so packing those preserves narrative context.
        if kind == "glossary":
            emit([group])
            continue
        # Emit-before-append: if this group would push the batch past target,
        # close the batch first. (Checking *after* appending — the tempting
        # one-liner — lets a near-full batch swallow a large group and produce
        # chunks of target + group size, blowing straight past max_chars.)
        if batch and batch_size + len(group) > target_chars:
            emit(batch)
            batch, batch_size = [], 0
        batch.append(group)
        batch_size += len(group)
    if batch:
        emit(batch)
    return chunks


_GLOSSARY_NUM = re.compile(r"^[0-9][0-9.\-]*[a-z]?\s+")


def _glossary_term(rule_text: str) -> str:
    """'1.-1.3 Ability\\nAn object property...' -> 'Ability'."""
    first_line = rule_text.split("\n", 1)[0]
    return _GLOSSARY_NUM.sub("", first_line).strip()


def chunk_chapter(
    html: str,
    slug: str,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[RuleChunk]:
    """Parse one chapter page and chunk it section by section."""
    kind: Literal["rule", "glossary"] = "glossary" if slug == "glossary" else "rule"
    blocks = parse_chapter(html)

    chunks: list[RuleChunk] = []
    chapter = section = ""
    groups: list[_RuleGroup] = []

    def flush() -> None:
        nonlocal groups
        if groups:
            chunks.extend(_chunks_from_section(
                groups, kind=kind, chapter=chapter, section=section or chapter,
                slug=slug, target_chars=target_chars, max_chars=max_chars,
            ))
            groups = []

    for block in blocks:
        if block.kind == "h1":
            flush()
            chapter, section = block.text, ""
        elif block.kind == "h2":
            flush()
            section = block.text
        elif block.kind == "rule":
            groups.append(_RuleGroup(block))
        elif block.kind == "subrule" and groups:
            groups[-1].subrules.append(block)
    flush()
    return chunks


# =============================================================================
# Loaders
# =============================================================================
def load_rule_chunks(
    html_dir: Path = RULES_HTML_DIR,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[RuleChunk]:
    """Chunk every CR chapter (run scripts/fetch_rules.py first)."""
    chunks: list[RuleChunk] = []
    for slug in CHAPTER_SLUGS:
        path = html_dir / f"{slug}.html"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `uv run python scripts/fetch_rules.py` first"
            )
        html = path.read_text(encoding="utf-8")
        chunks.extend(chunk_chapter(html, slug, target_chars=target_chars, max_chars=max_chars))
    return chunks


def load_keyword_chunks(path: Path = RAW_KEYWORD_FILE) -> list[RuleChunk]:
    """The card dataset's keyword glossary: 77 concise, reminder-text-style
    definitions. They overlap CR chapter 8 in topic but not in register — for
    "what does Dominate mean?" a one-liner often beats three paragraphs of
    tournament-grade precision. Retrieval gets to choose."""
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        RuleChunk(
            chunk_id=f"kw-{r['name'].lower().replace(' ', '-')}",
            kind="keyword",
            chapter="Keyword glossary",
            section="Keyword glossary",
            rule_ids=[],
            title=r["name"],
            text=expand_symbols(f"{r['name']}\n{r['description_plain']}"),
        )
        for r in records
        if r.get("description_plain")
    ]


def load_all_chunks() -> list[RuleChunk]:
    return load_rule_chunks() + load_keyword_chunks()


if __name__ == "__main__":
    # Self-check: chunk the corpus and report the size distribution — the
    # numbers that make "chunk size is a tuning knob" concrete.
    chunks = load_all_chunks()
    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    sizes = sorted(len(c.text) for c in chunks)
    n = len(sizes)
    print(f"{n} chunks  {by_kind}")
    print(f"chars: min={sizes[0]}  p50={sizes[n // 2]}  p90={sizes[9 * n // 10]}  max={sizes[-1]}")
    split_parts = [c for c in chunks if "/p" in c.chunk_id]
    print(f"oversized rules split into parts: {len(split_parts)}")

    sample = next(c for c in chunks if "cr7.0.3" in c.rule_ids)
    print(f"\n--- sample chunk {sample.chunk_id} ({sample.citation}) ---")
    print(f"breadcrumb: {sample.chapter} > {sample.section}")
    print(f"rule_ids:   {sample.rule_ids}")
    print(f"url:        {sample.source_url}")
    print(sample.text[:600])
