"""Hybrid retrieval — Phase 2, where embeddings become *search*.

This is the heart of the RAG pipeline. It combines the two halves the data
layer deliberately kept apart (see cards.py):

  - STRUCTURED filtering  — exact boolean predicates over typed metadata
    (color, pitch, class, legality). Answers "is this a legal blue Wizard
    card?" A card either matches or it's gone.
  - SEMANTIC ranking      — cosine similarity in embedding space. Answers
    "is this card *about* dealing arcane damage?" Returns a ranked list.

Neither alone is enough. We use **filter-then-rank**: shrink the corpus with
hard constraints first, then order the survivors by meaning. Illegal cards can
never leak into results (the filter is absolute), and among legal cards the
most semantically relevant float to the top.

Performance note: the corpus is small (~4.3k cards). The whole embedding index
is a ~12.5 MB float32 matrix that lives in RAM, and a query is a single
matrix-vector multiply — sub-millisecond. Because embeddings.py L2-normalizes
every vector, that dot product *is* cosine similarity. At this scale a
brute-force scan beats any approximate-nearest-neighbor index (FAISS et al.);
those earn their keep at millions of vectors, not thousands.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cards import Card, load_cards
from .embeddings import EMBED_DIM, EMBED_MODEL, embed_documents, embed_query

# data/index/card_index.npz, resolved relative to this file (-> repo root).
DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "index" / "card_index.npz"


# =============================================================================
# The structured half: metadata filters
# =============================================================================
@dataclass(frozen=True)
class CardFilter:
    """Hard constraints applied BEFORE semantic ranking.

    Every field is optional; an unset field (None / empty) imposes no
    constraint. List-valued fields are *any-of* (the card matches if it has at
    least one of the requested values). Scalar fields are exact-match.

    These are pure metadata predicates — deliberately *not* deck-construction
    rules (e.g. "a Wizard hero may also play Generic cards"). That higher-level
    legality logic belongs to a later layer; here we only filter on facts the
    card itself carries, keeping the structured half simple and predictable.
    """

    color: str | None = None              # "Red" / "Yellow" / "Blue"
    pitch: int | None = None              # exact pitch value
    cost: int | None = None               # exact resource cost
    classes: Sequence[str] = field(default_factory=tuple)      # any-of, e.g. ("Wizard",)
    talents: Sequence[str] = field(default_factory=tuple)      # any-of, e.g. ("Ice",)
    categories: Sequence[str] = field(default_factory=tuple)   # any-of, e.g. ("Attack",)
    keywords: Sequence[str] = field(default_factory=tuple)     # any-of, e.g. ("Go Again",)
    traits: Sequence[str] = field(default_factory=tuple)       # any-of
    legal_in: str | None = None           # format name, e.g. "cc" / "Blitz" / "Living Legend"

    def is_empty(self) -> bool:
        """True when no constraint is set (so we can skip the filter pass)."""
        return not any((
            self.color, self.pitch is not None, self.cost is not None,
            self.classes, self.talents, self.categories,
            self.keywords, self.traits, self.legal_in,
        ))

    def matches(self, card: Card) -> bool:
        """Return True if `card` satisfies every set constraint."""
        if self.color is not None and (card.color or "").lower() != self.color.lower():
            return False
        if self.pitch is not None and card.pitch != self.pitch:
            return False
        if self.cost is not None and card.cost != self.cost:
            return False
        if self.classes and not _any_of(self.classes, card.classes):
            return False
        if self.talents and not _any_of(self.talents, card.talents):
            return False
        if self.categories and not _any_of(self.categories, card.categories):
            return False
        if self.keywords and not _any_of(self.keywords, card.keywords):
            return False
        if self.traits and not _any_of(self.traits, card.traits):
            return False
        if self.legal_in is not None and not card.legality.is_legal(self.legal_in):
            return False
        return True


def _any_of(wanted: Sequence[str], have: Sequence[str]) -> bool:
    """Case-insensitive 'do these share at least one value?'"""
    have_lower = {h.lower() for h in have}
    return any(w.lower() in have_lower for w in wanted)


# =============================================================================
# A search result
# =============================================================================
@dataclass(frozen=True)
class SearchResult:
    card: Card
    score: float  # cosine similarity in [-1, 1]; higher = more relevant


# =============================================================================
# The retriever: corpus matrix + hybrid search
# =============================================================================
class Retriever:
    """Holds the embedded corpus and answers hybrid queries.

    Build it once (embedding ~4.3k cards takes a moment against Ollama), cache
    it to disk, and reuse. Prefer `Retriever.load_or_build()` — it transparently
    reloads the cache and only re-embeds when the underlying card text changes.
    """

    def __init__(self, cards: list[Card], vectors: np.ndarray) -> None:
        if vectors.shape != (len(cards), EMBED_DIM):
            raise ValueError(
                f"vectors {vectors.shape} don't match {len(cards)} cards x {EMBED_DIM} dims"
            )
        self.cards = cards
        self.vectors = vectors  # (n, EMBED_DIM), float32, unit-normalized

    # ---- search --------------------------------------------------------------
    def search(
        self, query: str, k: int = 10, filters: CardFilter | None = None
    ) -> list[SearchResult]:
        """Hybrid search: filter the corpus, then rank survivors by meaning.

        Returns up to `k` results, most relevant first.
        """
        # 1. STRUCTURED: which card rows survive the hard constraints?
        if filters is None or filters.is_empty():
            rows = np.arange(len(self.cards))
        else:
            rows = np.array(
                [i for i, c in enumerate(self.cards) if filters.matches(c)],
                dtype=np.intp,
            )
        if rows.size == 0:
            return []  # nothing matched the filter — semantic search has nothing to rank

        # 2. SEMANTIC: cosine similarity == dot product (vectors are normalized).
        q = embed_query(query)                  # (EMBED_DIM,)
        scores = self.vectors[rows] @ q         # (len(rows),)

        # 3. Top-k: argpartition finds the k best in O(n) without a full sort,
        #    then we sort just those k descending.
        k = min(k, rows.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchResult(self.cards[rows[i]], float(scores[i])) for i in top]

    # ---- build / persist -----------------------------------------------------
    @classmethod
    def build(cls, cards: list[Card]) -> "Retriever":
        """Embed every card's `text_for_embedding` into the corpus matrix."""
        texts = [c.text_for_embedding for c in cards]
        vectors = embed_documents(texts)
        return cls(cards, vectors)

    def save(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        """Persist the matrix + card ids + a fingerprint for cache validation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self.vectors,
            ids=np.array([c.unique_id for c in self.cards]),
            fingerprint=np.array(_fingerprint(self.cards)),
        )

    @classmethod
    def load_or_build(
        cls, cards: list[Card] | None = None, path: Path = DEFAULT_INDEX_PATH
    ) -> "Retriever":
        """Reload the cached index if it still matches the data, else rebuild.

        The cache is keyed on a fingerprint of the embedding text + model, so any
        change to a card's text (or a model swap) invalidates it automatically.
        """
        cards = cards if cards is not None else load_cards()
        fp = _fingerprint(cards)

        if path.exists():
            data = np.load(path, allow_pickle=False)
            ids = [str(x) for x in data["ids"]]
            if str(data["fingerprint"]) == fp and ids == [c.unique_id for c in cards]:
                return cls(cards, data["vectors"])

        retriever = cls.build(cards)
        retriever.save(path)
        return retriever


def _fingerprint(cards: list[Card]) -> str:
    """A content hash over the model + each card's id and embedding text.

    If any of these change, the cached vectors are stale and we re-embed.
    """
    h = hashlib.sha256()
    h.update(EMBED_MODEL.encode())
    h.update(str(EMBED_DIM).encode())
    for c in cards:
        h.update(c.unique_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(c.text_for_embedding.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
