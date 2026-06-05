"""Hybrid retrieval — Phase 2's core, generalized in Phase 4 to many sources.

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

Phase 4 makes the corpus HETEROGENEOUS: rules text (rules.py) joins the cards.
The key move is the `Document` protocol — the Retriever stops knowing what a
Card is and only requires "an id and text to embed". Everything card-specific
(CardFilter, hero predicates) moves UP into `scope()`, which compiles those
constraints into one Document-level predicate the source-agnostic search hook
accepts. Each source keeps its own cached index (a rules tweak shouldn't
re-embed 4.3k cards); `Retriever.merge()` stacks them into one searchable
corpus, which works *because* every source lives in the same embedding space —
one model, one geometry, so a question can land near a card or a rule alike.

Performance note: the corpus is small (~4.3k cards + ~600 rule chunks). The
whole embedding index is a ~14 MB float32 matrix that lives in RAM, and a
query is a single matrix-vector multiply — sub-millisecond. Because
embeddings.py L2-normalizes every vector, that dot product *is* cosine
similarity. At this scale a brute-force scan beats any approximate-nearest-
neighbor index (FAISS et al.); those earn their keep at millions of vectors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .cards import Card, load_cards
from .embeddings import EMBED_DIM, EMBED_MODEL, embed_documents, embed_query
from .rules import RuleChunk, load_all_chunks

# Per-source indexes under data/index/, resolved relative to this file.
_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
CARD_INDEX_PATH = _INDEX_DIR / "card_index.npz"
RULES_INDEX_PATH = _INDEX_DIR / "rules_index.npz"

Source = Literal["cards", "rules", "all"]


# =============================================================================
# The Document protocol: what it takes to be retrievable
# =============================================================================
@runtime_checkable
class Document(Protocol):
    """Anything with a stable id and text to embed can join the corpus.

    This is a typing.Protocol — *structural* typing. Card and RuleChunk never
    inherit from it or import it; they conform simply by having these two
    properties. The Retriever depends on this minimal surface and nothing
    else, which is exactly what lets it stay ignorant of card colors and rule
    numbers alike.
    """

    @property
    def doc_id(self) -> str: ...

    @property
    def text_for_embedding(self) -> str: ...


# =============================================================================
# The structured half: metadata filters (card-specific, applied via scope())
# =============================================================================
@dataclass(frozen=True)
class CardFilter:
    """Hard constraints applied BEFORE semantic ranking.

    Every field is optional; an unset field (None / empty) imposes no
    constraint. List-valued fields are *any-of* (the card matches if it has at
    least one of the requested values). Scalar fields are exact-match.

    These are pure metadata predicates — deliberately *not* deck-construction
    rules (e.g. "a Wizard hero may also play Generic cards"). That higher-level
    legality logic belongs to deck.py; here we only filter on facts the card
    itself carries, keeping the structured half simple and predictable.
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


def scope(
    source: Source = "all",
    filters: CardFilter | None = None,
    card_predicate: Callable[[Card], bool] | None = None,
) -> Callable[[Document], bool] | None:
    """Compile mixed-corpus constraints into ONE Document-level predicate.

    The semantics encode a deliberate choice: card constraints CONSTRAIN
    CARDS, they don't exclude rules. Asking "--class Wizard, how does arcane
    damage work?" should narrow the card side to Wizard cards while still
    letting the arcane-damage rules through — a card filter says what kind of
    cards you want, not that you suddenly stopped wanting rules. What *does*
    gate by type is `source` ("cards" / "rules" / "all").

    Returns None when nothing constrains anything, so search() can skip the
    per-document Python loop entirely (the fast path).
    """
    has_filters = filters is not None and not filters.is_empty()
    if source == "all" and not has_filters and card_predicate is None:
        return None

    def predicate(doc: Document) -> bool:
        if isinstance(doc, Card):
            if source == "rules":
                return False
            if has_filters and not filters.matches(doc):
                return False
            return card_predicate is None or card_predicate(doc)
        # Non-card (rules text): only the source gate applies.
        return source != "cards"

    return predicate


# =============================================================================
# A search result
# =============================================================================
@dataclass(frozen=True)
class SearchResult:
    doc: Document  # a Card or a RuleChunk — isinstance() to tell, or duck-type
    score: float   # cosine similarity in [-1, 1]; higher = more relevant


# =============================================================================
# The retriever: corpus matrix + hybrid search
# =============================================================================
class Retriever:
    """Holds an embedded corpus of Documents and answers hybrid queries.

    Build it once (embedding takes a moment against Ollama), cache it to disk,
    and reuse. Prefer `load_or_build()` — it transparently reloads the cache
    and only re-embeds when the underlying document text changes.
    """

    def __init__(self, docs: Sequence[Document], vectors: np.ndarray) -> None:
        if vectors.shape != (len(docs), EMBED_DIM):
            raise ValueError(
                f"vectors {vectors.shape} don't match {len(docs)} docs x {EMBED_DIM} dims"
            )
        self.docs = list(docs)
        self.vectors = vectors  # (n, EMBED_DIM), float32, unit-normalized

    # ---- search --------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 10,
        predicate: Callable[[Document], bool] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: filter the corpus, then rank survivors by meaning.

        `predicate` is the single structured-filtering hook — build one with
        scope() to combine source selection, CardFilters, and deck-legality
        tests. Returns up to `k` results, most relevant first.
        """
        # 1. STRUCTURED: which corpus rows survive the hard constraints?
        if predicate is None:
            rows = np.arange(len(self.docs))
        else:
            rows = np.array(
                [i for i, d in enumerate(self.docs) if predicate(d)], dtype=np.intp
            )
        if rows.size == 0:
            return []  # nothing matched the filter — nothing to rank

        # 2. SEMANTIC: cosine similarity == dot product (vectors are normalized).
        q = embed_query(query)                  # (EMBED_DIM,)
        scores = self.vectors[rows] @ q         # (len(rows),)

        # 3. Top-k: argpartition finds the k best in O(n) without a full sort,
        #    then we sort just those k descending.
        k = min(k, rows.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchResult(self.docs[rows[i]], float(scores[i])) for i in top]

    # ---- build / persist -----------------------------------------------------
    @classmethod
    def build(cls, docs: Sequence[Document]) -> "Retriever":
        """Embed every document's `text_for_embedding` into the corpus matrix."""
        texts = [d.text_for_embedding for d in docs]
        vectors = embed_documents(texts)
        return cls(docs, vectors)

    def save(self, path: Path) -> None:
        """Persist the matrix + doc ids + a fingerprint for cache validation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self.vectors,
            ids=np.array([d.doc_id for d in self.docs]),
            fingerprint=np.array(_fingerprint(self.docs)),
        )

    @classmethod
    def load_or_build(cls, docs: Sequence[Document], path: Path) -> "Retriever":
        """Reload the cached index if it still matches the data, else rebuild.

        The cache is keyed on a fingerprint of the embedding text + model, so
        any change to a document's text (or a model swap) invalidates it
        automatically.
        """
        fp = _fingerprint(docs)
        if path.exists():
            data = np.load(path, allow_pickle=False)
            ids = [str(x) for x in data["ids"]]
            if str(data["fingerprint"]) == fp and ids == [d.doc_id for d in docs]:
                return cls(docs, data["vectors"])

        retriever = cls.build(docs)
        retriever.save(path)
        return retriever

    @classmethod
    def merge(cls, *parts: "Retriever") -> "Retriever":
        """Stack per-source retrievers into one searchable corpus.

        Sound because every part was embedded by the same model into the same
        space — similarity scores between a query and a card vs. a rule chunk
        are directly comparable. (If sources ever used different embedding
        models, this would be silently meaningless; the EMBED_DIM check in
        __init__ catches dimension mismatches but not model mismatches —
        that's what the per-index fingerprints guard.)
        """
        docs = [d for p in parts for d in p.docs]
        vectors = np.vstack([p.vectors for p in parts])
        return cls(docs, vectors)


def _fingerprint(docs: Sequence[Document]) -> str:
    """A content hash over the model + each document's id and embedding text.

    If any of these change, the cached vectors are stale and we re-embed.
    """
    h = hashlib.sha256()
    h.update(EMBED_MODEL.encode())
    h.update(str(EMBED_DIM).encode())
    for d in docs:
        h.update(d.doc_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(d.text_for_embedding.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# =============================================================================
# Default corpus builders
# =============================================================================
def card_retriever() -> Retriever:
    """The card corpus, from its own cached index."""
    return Retriever.load_or_build(load_cards(), CARD_INDEX_PATH)


def rules_retriever() -> Retriever:
    """The rules-text corpus (CR chunks + glossaries), from its own index."""
    return Retriever.load_or_build(load_all_chunks(), RULES_INDEX_PATH)


def corpus_retriever() -> Retriever:
    """Everything: cards + rules in one searchable space."""
    return Retriever.merge(card_retriever(), rules_retriever())
