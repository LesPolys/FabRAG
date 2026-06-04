"""RAG orchestration — Phase 3, wiring the pipeline end to end.

This is the thin layer that turns the three building blocks into one answer:

    question
      -> Retriever.search()        (retrieval.py: filter + semantic rank)
      -> format_context()          (here: render cards as grounding text)
      -> generate()                (generation.py: local LLM, grounded reply)

A note on what the context includes. text_for_embedding (cards.py) deliberately
*excludes* numeric stats — embeddings reason poorly about exact numbers, so
pitch/cost/power live in the structured filter instead. Generation is the
opposite: the LLM needs those facts to answer "what's a cheap arcane attack?",
so format_context() *adds the stats back in*. Same cards, different projection
for a different consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import expand_symbols
from .generation import generate, generate_stream
from .retrieval import CardFilter, Retriever, SearchResult

# A retrieval result with no cards needs no LLM call — we answer directly.
_NO_MATCH_MSG = (
    "No cards matched your query and filters, so there's nothing to answer from. "
    "Try loosening the filters or rephrasing."
)


@dataclass(frozen=True)
class RagResponse:
    answer: str
    results: list[SearchResult]  # the cards retrieved, for citation/inspection


# --- context formatting ------------------------------------------------------
def _format_card(card) -> str:
    """Render one card as grounding text — name, type line, STATS, and rules.

    Stats are included here (unlike the embedding projection) because the model
    needs them to reason about cost/power/etc. Symbols are expanded to words so
    the LLM reads "Gain 3 life", not "Gain 3{h}".
    """
    lines = [card.name, card.type_text]

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


def format_context(results: list[SearchResult]) -> str:
    """Render retrieved cards into the numbered CARDS block the LLM sees."""
    return "\n\n".join(
        f"[{i}] {_format_card(r.card)}" for i, r in enumerate(results, start=1)
    )


# --- the pipeline ------------------------------------------------------------
_default_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Lazily build/load the shared corpus index once, then reuse it.

    Building embeds ~4.3k cards (slow); we never want to do it per question.
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever.load_or_build()
    return _default_retriever


def _retrieve(
    question: str, k: int, filters: CardFilter | None, retriever: Retriever | None
) -> tuple[list[SearchResult], str]:
    retriever = retriever if retriever is not None else get_retriever()
    results = retriever.search(question, k=k, filters=filters)
    return results, format_context(results)


def answer(
    question: str,
    *,
    k: int = 8,
    filters: CardFilter | None = None,
    retriever: Retriever | None = None,
) -> RagResponse:
    """Full RAG: retrieve cards for `question`, then generate a grounded answer."""
    results, context = _retrieve(question, k, filters, retriever)
    if not results:
        return RagResponse(_NO_MATCH_MSG, [])
    return RagResponse(generate(question, context), results)


def answer_stream(
    question: str,
    *,
    k: int = 8,
    filters: CardFilter | None = None,
    retriever: Retriever | None = None,
):
    """Like answer(), but return (results, token_iterator) for live CLI output.

    Results come back immediately (retrieval is fast) so the caller can show
    citations while the answer streams in.
    """
    results, context = _retrieve(question, k, filters, retriever)
    if not results:
        return [], iter([_NO_MATCH_MSG])
    return results, generate_stream(question, context)
