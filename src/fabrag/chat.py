"""Deck chat — Phase 10, from "generate a deck" to "work on a deck".

Phase 6 built a deck in one agentic pass (generate -> validate -> repair). This
is the conversational sequel: a multi-turn assistant grounded in ONE specific
deck plus the rules corpus. The pilot can ask "why is this card here?", "what
beats aggro?", "swap something in for more defense" — and every turn the agent
retrieves the relevant rules and HERO-LEGAL cards, so its answers cite real
rules and only ever suggest cards the deck could actually run.

It reuses the whole stack rather than re-inventing it:
  - retrieval (rag.retrieve) with the hero-pool predicate, so suggested cards
    are legal for this hero — the same guard build.py's candidate sourcing uses;
  - the grounded-answer contract (generation.py) extended to multi-turn via
    chat_stream: the deck sits in the system prompt (static for the chat), prior
    turns carry the conversation, and each new turn appends a freshly retrieved
    CONTEXT block.

State lives with the caller (the deck + history are passed in each turn), so the
engine is stateless and both front doors — the CLI REPL and the web workspace —
drive it the same way.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

import ollama

from . import rag
from .cards import Card
from .deck import hero_pool_predicate
from .generation import CHAT_MODEL, DEFAULT_TEMPERATURE, NUM_CTX, chat_stream
from .pool import CardPool

DECK_CHAT_SYSTEM = """\
You are FabRAG, an expert Flesh and Blood deck-building assistant discussing ONE
specific deck with its pilot. You are given that deck below as DECK, and on each
turn a CONTEXT section of freshly retrieved rules and hero-legal cards.

Follow these rules without exception:
1. Ground every claim in the DECK, the CONTEXT, or the conversation so far. Do
   not rely on prior knowledge of Flesh and Blood — card text and rules change.
   If the data doesn't cover something, say so plainly.
2. When suggesting cards to add or swap in, name ONLY cards that appear in the
   CONTEXT (they are drawn from this hero's legal pool) or are already in the
   DECK. Never invent a card.
3. Cite rules by the exact citation shown (e.g. "CR 7.3.2", "CR Glossary:
   Dominate") and cards by their exact name.
4. Be concise and concrete — the pilot can see the decklist; talk strategy, not
   filler.

DECK:
{deck}
"""


def pool_to_text(pool: CardPool) -> str:
    """Render a CardPool as the compact decklist the chat is grounded in."""
    return deck_to_text(
        pool.hero.name,
        pool.format,
        _group(pool.deck),
        _group(pool.inventory),
        _group(pool.sideboard),
    )


def _group(cards: list[Card]) -> list[tuple[str, int | None, int]]:
    """Collapse a card list into sorted (name, pitch, copies) rows."""
    counts = Counter((c.name, c.pitch) for c in cards)
    return [(name, pitch, n) for (name, pitch), n in sorted(counts.items())]


def deck_to_text(
    hero_name: str,
    fmt: str,
    deck: list[tuple[str, int | None, int]],
    inventory: list[tuple[str, int | None, int]],
    sideboard: list[tuple[str, int | None, int]],
) -> str:
    """Render a decklist from grouped (name, pitch, copies) rows — the shared
    format both front doors feed the chat (the web sends rows; the CLI derives
    them from a CardPool via pool_to_text)."""
    def fmt_rows(rows: list[tuple[str, int | None, int]]) -> str:
        return "; ".join(
            f"{n}x {name}" + (f" (pitch {pitch})" if pitch is not None else "")
            for name, pitch, n in rows
        ) or "(none)"

    return (
        f"Hero: {hero_name}  |  Format: {fmt}\n"
        f"Inventory: {fmt_rows(inventory)}\n"
        f"Deck: {fmt_rows(deck)}\n"
        f"Sideboard: {fmt_rows(sideboard)}"
    )


def deck_chat_stream(
    hero: Card,
    fmt: str,
    deck_text: str,
    history: list[dict],
    question: str,
    *,
    k: int = 8,
    retriever=None,
):
    """One chat turn: retrieve grounding for `question`, then stream a reply.

    `history` is a list of {"role": "user"|"assistant", "content": str} from the
    conversation so far. Returns (results, token_iterator): the retrieved
    documents (so the caller can show citations) and the streamed answer.
    """
    predicate = hero_pool_predicate(hero, fmt)
    results = rag.retrieve(question, k=k, source="all", predicate=predicate, retriever=retriever)
    context = rag.format_context(results) if results else "(no cards or rules retrieved)"

    messages = [
        {"role": "system", "content": DECK_CHAT_SYSTEM.format(deck=deck_text)},
        *[{"role": t["role"], "content": t["content"]} for t in history],
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]
    return results, chat_stream(messages)


# --- tool-calling variant: let the model decide WHEN (and what) to search ----
# The always-retrieve turn above searches once per message whether or not it
# helps ("thanks" still fires a search). Here the model is handed a search tool
# and chooses: a chit-chat turn calls nothing, "what beats Briar?" calls it with
# a focused query. 7B tool-use is the risk, so deck_chat_agentic falls back to
# the proven path on any failure — tool-calling is an optimization, not a
# dependency.
_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_pool_and_rules",
        "description": (
            "Search THIS hero's legal card pool and the official rules corpus for "
            "cards or rules relevant to a query. Call it when you need specific "
            "cards (e.g. to suggest a swap) or exact rules text; skip it for "
            "small talk or questions the deck already answers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "what to look for, by effect — e.g. 'cards that prevent arcane damage'",
                }
            },
            "required": ["query"],
        },
    },
}

_TOOL_HINT = (
    "\nYou may call search_pool_and_rules to look up cards or rules before "
    "answering. Search only when it genuinely helps; otherwise just answer."
)


def deck_chat_agentic(
    hero: Card,
    fmt: str,
    deck_text: str,
    history: list[dict],
    question: str,
    *,
    k: int = 8,
    retriever=None,
    max_tool_rounds: int = 3,
):
    """Tool-calling chat turn: the model decides whether/what to search.

    Resolves any tool calls in a short non-streamed loop (each runs a hero-pool
    retrieval), then streams the final answer. If the model answers without
    searching, that reply is returned directly. Any tool-calling failure falls
    back to deck_chat_stream, so the chat never breaks on flaky 7B tool use.

    Returns (results, token_iterator) like deck_chat_stream — `results` is
    whatever the model chose to retrieve (possibly empty).
    """
    predicate = hero_pool_predicate(hero, fmt)
    messages: list = [
        {"role": "system", "content": DECK_CHAT_SYSTEM.format(deck=deck_text) + _TOOL_HINT},
        *[{"role": t["role"], "content": t["content"]} for t in history],
        {"role": "user", "content": question},
    ]
    grounding: list = []
    try:
        for _ in range(max_tool_rounds):
            resp = ollama.chat(
                model=CHAT_MODEL,
                messages=messages,
                tools=[_SEARCH_TOOL],
                options={"temperature": DEFAULT_TEMPERATURE, "num_ctx": NUM_CTX},
            )
            calls = resp.message.tool_calls or []
            if not calls:
                # Model answered without (further) searching — return its reply.
                content = resp.message.content or ""
                return grounding, iter([content] if content else [])
            messages.append(resp.message)  # preserve the tool_calls for the follow-up
            for tc in calls:
                args = dict(tc.function.arguments or {})
                query = str(args.get("query", "")).strip() or question
                results = rag.retrieve(
                    query, k=k, source="all", predicate=predicate, retriever=retriever
                )
                grounding.extend(results)
                body = rag.format_context(results) if results else "(no matching cards or rules)"
                messages.append({"role": "tool", "content": body})
        # Rounds exhausted while still searching — force a final answer (no tools).
        return grounding, chat_stream(messages)
    except Exception:
        # Flaky tool use, unsupported model, etc. — never break the chat.
        return deck_chat_stream(hero, fmt, deck_text, history, question, k=k, retriever=retriever)


if __name__ == "__main__":
    # Self-check: build a small deck once, then ask it one question (no REPL).
    from .build import build_deck

    result = build_deck("Ira, Crimson Haze", "fast aggressive ninja attacks", "blitz")
    pool = result.pool
    q = "Why are there defense reactions in this aggressive deck, and what could I add against arcane damage?"
    print(f"Q: {q}\n")
    _results, stream = deck_chat_agentic(pool.hero, pool.format, pool_to_text(pool), [], q)
    for piece in stream:
        print(piece, end="", flush=True)
    print()
