"""RAG orchestration — Phase 3, wiring the pipeline end to end.

This is the thin layer that turns the building blocks into one answer:

    question
      -> retrieve()                (per-source top-k over the mixed corpus)
      -> format_context()          (here: render cards AND rules as grounding text)
      -> generate()                (generation.py: local LLM, grounded reply)

A note on what the context includes. text_for_embedding (cards.py) deliberately
*excludes* numeric stats — embeddings reason poorly about exact numbers, so
pitch/cost/power live in the structured filter instead. Generation is the
opposite: the LLM needs those facts to answer "what's a cheap arcane attack?",
so format_context() *adds the stats back in*. Same cards, different projection
for a different consumer.

Phase 4 adds the rules corpus, and with it the central mixed-retrieval lesson:
similarity scores are NOT comparable across heterogeneous sources. Card text is
short and punchy; rule chunks are long and topic-diluted — measured on this
corpus, cards score ~0.03 hotter on *everything*, and at 4.3k cards vs 600
chunks the majority source floods any single top-k (the right rule for "what
happens when an attack is defended?" ranked 18th behind 17 incidental cards).
So retrieve() gives EACH SOURCE ITS OWN top-k quota and lets the context carry
both. Per-source quotas are the simplest of the standard fixes (others:
per-source score normalization, reciprocal rank fusion) — Phase 5's eval
harness is where we'd justify anything fancier.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .cards import Card, expand_symbols
from .generation import generate, generate_stream
from .retrieval import (
    CardFilter,
    Retriever,
    SearchResult,
    Source,
    corpus_retriever,
    scope,
)
from .rules import RuleChunk

# A retrieval result with no documents needs no LLM call — we answer directly.
_NO_MATCH_MSG = (
    "Nothing in the corpus matched your query and filters, so there's nothing "
    "to answer from. Try loosening the filters or rephrasing."
)


@dataclass(frozen=True)
class RagResponse:
    answer: str
    results: list[SearchResult]  # the documents retrieved, for citation/inspection


# --- context formatting ------------------------------------------------------
def _format_card(card: Card) -> str:
    """Render one card as grounding text — name, type line, STATS, and rules.

    Stats are included here (unlike the embedding projection) because the model
    needs them to reason about cost/power/etc. Symbols are expanded to words so
    the LLM reads "Gain 3 life", not "Gain 3{h}".
    """
    lines = [f"Card: {card.name}", card.type_text]

    stats = []
    if card.color:
        stats.append(f"Color {card.color}")
    if card.pitch is not None:
        stats.append(f"Pitch {card.pitch}")
    if card.cost is not None:
        stats.append(f"Cost {card.cost}")
    if card.power is not None:
        stats.append(f"Power {card.power}")
    if card.defense is not None:
        stats.append(f"Defense {card.defense}")
    if card.health is not None:
        stats.append(f"Health {card.health}")
    if card.intelligence is not None:
        stats.append(f"Intellect {card.intelligence}")
    if stats:
        lines.append(", ".join(stats))

    if card.keywords:
        lines.append("Keywords: " + ", ".join(card.keywords))
    if card.text:
        lines.append(expand_symbols(card.text))
    return "\n".join(lines)


def _format_rule(chunk: RuleChunk) -> str:
    """Render one rules chunk — citation header (so the LLM can cite "CR 7.3.2"
    back at us), location breadcrumb, then the rules text itself."""
    header = f"Rule {chunk.citation}"
    if chunk.kind == "rule":
        header += f" — {chunk.chapter} > {chunk.section}"
    return f"{header}\n{chunk.text}"


def format_context(results: list[SearchResult]) -> str:
    """Render retrieved documents into the numbered CONTEXT block the LLM sees.

    Each entry is rendered by type — cards with their stats restored, rules with
    their citation — under one continuous numbering, so the prompt's "cite your
    sources" contract works the same way for both.
    """
    parts = []
    for i, r in enumerate(results, start=1):
        body = _format_card(r.doc) if isinstance(r.doc, Card) else _format_rule(r.doc)
        parts.append(f"[{i}] {body}")
    return "\n\n".join(parts)


# --- the pipeline ------------------------------------------------------------
_default_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Lazily build/load the shared mixed corpus (cards + rules) once, reuse it.

    Building embeds ~4.9k documents (slow); we never want that per question.
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = corpus_retriever()
    return _default_retriever


def retrieve(
    question: str,
    *,
    k: int = 8,
    source: Source = "all",
    filters: CardFilter | None = None,
    predicate: Callable[[Card], bool] | None = None,
    retriever: Retriever | None = None,
) -> list[SearchResult]:
    """Retrieve up to `k` documents for `question`, quota-ed per source.

    For source="all" we deliberately do NOT take one top-k over the merged
    corpus — scores aren't comparable across sources (see module docstring).
    Instead the budget splits ~70/30 cards/rules, each side ranked within its
    own source. The split is a guess; Phase 5's eval exists to tune it.

    Results come back cards-first then rules — context-block order, not global
    score order (interleaving by raw score would re-import the bias the quotas
    just removed).
    """
    retriever = retriever if retriever is not None else get_retriever()
    if source != "all":
        return retriever.search(
            question, k=k, predicate=scope(source, filters, predicate)
        )
    k_rules = max(2, round(k * 0.3))
    k_cards = max(1, k - k_rules)
    cards = retriever.search(
        question, k=k_cards, predicate=scope("cards", filters, predicate)
    )
    rules = retriever.search(question, k=k_rules, predicate=scope("rules"))
    return cards + rules


def _retrieve_and_format(
    question: str,
    k: int,
    source: Source,
    filters: CardFilter | None,
    predicate: Callable[[Card], bool] | None,
    retriever: Retriever | None,
) -> tuple[list[SearchResult], str]:
    results = retrieve(
        question, k=k, source=source, filters=filters,
        predicate=predicate, retriever=retriever,
    )
    return results, format_context(results)


def answer(
    question: str,
    *,
    k: int = 8,
    source: Source = "all",
    filters: CardFilter | None = None,
    predicate: Callable[[Card], bool] | None = None,
    retriever: Retriever | None = None,
) -> RagResponse:
    """Full RAG: retrieve context for `question`, then generate a grounded answer.

    `predicate` (e.g. deck.hero_pool_predicate) restricts the card side to a
    hero's legal pool, so the answer only ever suggests deck-legal cards;
    `source` selects which corpora ground the answer.
    """
    results, context = _retrieve_and_format(question, k, source, filters, predicate, retriever)
    if not results:
        return RagResponse(_NO_MATCH_MSG, [])
    return RagResponse(generate(question, context), results)


def answer_stream(
    question: str,
    *,
    k: int = 8,
    source: Source = "all",
    filters: CardFilter | None = None,
    predicate: Callable[[Card], bool] | None = None,
    retriever: Retriever | None = None,
):
    """Like answer(), but return (results, token_iterator) for live CLI output.

    Results come back immediately (retrieval is fast) so the caller can show
    citations while the answer streams in.
    """
    results, context = _retrieve_and_format(question, k, source, filters, predicate, retriever)
    if not results:
        return [], iter([_NO_MATCH_MSG])
    return results, generate_stream(question, context)
