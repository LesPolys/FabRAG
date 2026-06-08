"""The web UI — Phase 7, a thin HTTP skin over the existing pipeline.

Architecture rule: NO logic lives here. Every route is parse-request →
call the same functions the CLI calls (rag.retrieve, rag.answer_stream,
build.build_deck) → serialize to JSON. If a behavior needs changing, it
changes in the pipeline module and both front doors (CLI + web) get it.

The one web-specific concern is STREAMING: fabrag ask streams tokens in the
terminal, and the browser should get the same liveness. /api/ask uses
Server-Sent Events (SSE) — the simplest browser-native streaming transport:
a long-lived HTTP response of "event:/data:" lines that EventSource consumes
with reconnection for free. (WebSockets would be overkill: traffic is
strictly server -> client.)

Run:  uv run fabrag serve   ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import rag
from ..cards import Card, expand_symbols
from ..retrieval import CardFilter, SearchResult
from ..rules import RuleChunk

app = FastAPI(title="FabRAG")

_STATIC = Path(__file__).parent / "static"


# =============================================================================
# Serializers: pipeline objects -> JSON the frontend renders
# =============================================================================
def _card_json(c: Card, score: float | None = None) -> dict:
    return {
        "kind": "card",
        "name": c.name,
        "type_text": c.type_text,
        "image_url": c.image_url,
        "color": c.color,
        "pitch": c.pitch,
        "cost": c.cost,
        "power": c.power,
        "defense": c.defense,
        "text": expand_symbols(c.text) if c.text else "",
        "horizontal": c.played_horizontally,
        "score": score,
    }


def _rule_json(r: RuleChunk, score: float | None = None) -> dict:
    return {
        "kind": "rule",
        "citation": r.citation,
        "chapter": r.chapter,
        "section": r.section,
        "url": r.source_url,
        "text": r.text,
        "score": score,
    }


def _result_json(r: SearchResult) -> dict:
    if isinstance(r.doc, Card):
        return _card_json(r.doc, r.score)
    return _rule_json(r.doc, r.score)


def _hero_predicate(hero_name: str | None, fmt: str):
    """Resolve an optional hero name into a pool predicate (or None)."""
    if not hero_name:
        return None
    from ..deck import find_hero, hero_pool_predicate

    cards = [d for d in rag.get_retriever().docs if isinstance(d, Card)]
    hero = find_hero(hero_name, cards)  # ValueError -> 400 below
    return hero_pool_predicate(hero, fmt)


# =============================================================================
# Routes
# =============================================================================
# Result-set cap: ranking is a single matrix op over ~4.9k docs, so a deep
# result set costs almost nothing server-side — the cap bounds memory/JSON.
_MAX_DEPTH = 500

# Sort options for the card list. Each maps to (key, descending). None values
# (a card with no cost, say) always sort to the end regardless of direction.
_SORTS = {
    "name": (lambda c: c.name.lower(), False),
    "pitch": (lambda c: c.pitch, False),
    "cost": (lambda c: c.cost, False),
    "power": (lambda c: c.power, True),
    "defense": (lambda c: c.defense, True),
}


def _sorted_cards(results: list[SearchResult], sort: str) -> list[SearchResult]:
    key, desc = _SORTS[sort]

    def sort_key(r: SearchResult):
        v = key(r.doc)
        if v is None:
            return (1, 0)
        return (0, -v if (desc and not isinstance(v, str)) else v)

    return sorted(results, key=sort_key)


@app.get("/api/search")
def api_search(
    q: str,
    k: int = 24,
    page: int = 1,
    sort: str = "relevance",
    source: str = "cards",
    color: str | None = None,
    pitch: int | None = None,
    card_class: str | None = None,
    legal: str | None = None,
    hero: str | None = None,
):
    """Ranked search with numbered pages and optional card sorting.

    Shape: the CARD list is the pageable/sortable thing (sort over the FULL
    capped result set, then slice — page 1 of "cost ascending" really is the
    cheapest matches overall). Rule hits don't sort by power, so they come
    back as a separate relevance-ordered list: paged when source=rules,
    pinned top-few when source=all. Re-ranking per request is fine — scoring
    the whole corpus is sub-millisecond; page-state caching would be
    machinery without a payoff.
    """
    if sort != "relevance" and sort not in _SORTS:
        raise HTTPException(status_code=400, detail=f"unknown sort {sort!r}")
    filters = CardFilter(
        color=color,
        pitch=pitch,
        classes=(card_class,) if card_class else (),
        legal_in=legal,
    )
    try:
        predicate = _hero_predicate(hero, legal or "cc")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    page = max(1, page)
    lo, hi = (page - 1) * k, page * k

    def fetch(src: str, depth: int) -> list[SearchResult]:
        return rag.retrieve(q, k=depth, source=src, filters=filters, predicate=predicate)

    cards: list[SearchResult] = []
    rules: list[SearchResult] = []
    if source in ("cards", "all"):
        cards = fetch("cards", _MAX_DEPTH)
        if sort != "relevance":
            cards = _sorted_cards(cards, sort)
    if source == "all":
        rules = fetch("rules", 4)  # pinned context, not the main list
    elif source == "rules":
        rules = fetch("rules", _MAX_DEPTH)

    paged = cards if source != "rules" else rules
    total = len(paged)
    return {
        "cards": [_result_json(r) for r in (cards[lo:hi] if source != "rules" else [])],
        "rules": [_result_json(r) for r in (rules[lo:hi] if source == "rules" else rules)],
        "total": total,
        "page": page,
        "pages": max(1, -(-total // k)),  # ceil
    }


@app.get("/api/ask")
def api_ask(q: str, k: int = 8, source: str = "all", hero: str | None = None):
    """SSE stream: one 'grounding' event (the retrieved docs, so the UI can
    show citations immediately), then 'token' events, then 'done'."""
    try:
        predicate = _hero_predicate(hero, "cc")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def stream():
        results, tokens = rag.answer_stream(q, k=k, source=source, predicate=predicate)
        grounding = json.dumps({"results": [_result_json(r) for r in results]})
        yield f"event: grounding\ndata: {grounding}\n\n"
        for t in tokens:
            yield f"event: token\ndata: {json.dumps({'t': t})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


class ChatRequest(BaseModel):
    hero: str
    format: str = "cc"
    deck: list[dict] = []        # grouped rows: {name, pitch, copies}
    inventory: list[dict] = []
    sideboard: list[dict] = []
    history: list[dict] = []     # prior turns: {role, content}
    message: str


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """Multi-turn deck chat over POST (the deck + history don't fit a query
    string). Same SSE shape as /api/ask — a 'grounding' event then 'token's —
    but read by the frontend with fetch+reader, since POST rules out EventSource.

    Stateless: the client holds the deck and history and replays them each turn,
    so the server reconstructs only what grounding needs — the hero (for the
    pool predicate) and a textual decklist (from the grouped rows sent)."""
    from .. import chat
    from ..deck import find_hero

    cards = [d for d in rag.get_retriever().docs if isinstance(d, Card)]
    try:
        hero = find_hero(req.hero, cards)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def rows(entries):
        return [(e.get("name", ""), e.get("pitch"), e.get("copies", 1)) for e in entries]

    deck_text = chat.deck_to_text(
        hero.name, req.format, rows(req.deck), rows(req.inventory), rows(req.sideboard)
    )

    def stream():
        results, tokens = chat.deck_chat_stream(
            hero, req.format, deck_text, req.history, req.message
        )
        grounding = json.dumps({"results": [_result_json(r) for r in results]})
        yield f"event: grounding\ndata: {grounding}\n\n"
        for t in tokens:
            yield f"event: token\ndata: {json.dumps({'t': t})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


class BuildRequest(BaseModel):
    hero: str
    strategy: str = "a balanced, efficient deck"
    format: str = "cc"
    explain: bool = True


@app.post("/api/build")
def api_build(req: BuildRequest):
    """Synchronous on purpose: a build is 1-3 minutes of LLM rounds, and a
    spinner with honest expectations beats premature job-queue machinery."""
    from ..build import build_deck, deck_context
    from ..deck_quality import score_deck
    from ..formats import get_format
    from ..stats import deck_stats

    try:
        result = build_deck(req.hero, req.strategy, req.format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pool = result.pool

    explanation = ""
    if req.explain:
        from ..generation import generate

        question = (
            f"This {pool.format} deck was built for {pool.hero.name} "
            f"with the strategy: {req.strategy!r}. Explain the game plan in 2-3 "
            f"short paragraphs, citing cards by name."
        )
        explanation = generate(question, deck_context(result))

    def _grouped(cards):
        """One entry per (name, pitch) with a count — the display unit."""
        from collections import Counter

        counts = Counter((c.name, c.pitch) for c in cards)
        by_key = {(c.name, c.pitch): c for c in cards}
        out = []
        for (name, pitch), n in sorted(counts.items()):
            entry = _card_json(by_key[(name, pitch)])
            entry["copies"] = n
            out.append(entry)
        return out

    inventory = []
    for c in sorted(pool.inventory, key=lambda c: (c.equipment_slot or "Weapon", c.name)):
        entry = _card_json(c)
        entry["copies"] = 1
        entry["slot"] = c.equipment_slot or "Weapon"
        inventory.append(entry)

    # Sideboard entries carry the matchup rationale parsed from the build notes.
    # A note reads "+ Name (pitch N) — reason; for X"; key it by Name alone (the
    # pitch suffix would otherwise never match the entry's name).
    def _note_name(note: str) -> str:
        head = note.lstrip("+ ").split(" — ", 1)[0]
        return head.split(" (pitch ", 1)[0].strip()

    sideboard = _grouped(pool.sideboard)
    reasons = {_note_name(n): n.split(" — ", 1)[1] for n in result.sideboard_notes if " — " in n}
    for entry in sideboard:
        entry["reason"] = reasons.get(entry["name"], "")

    s = deck_stats(pool)
    # JSON object keys are strings; ints would round-trip as strings anyway, so
    # make the conversion explicit and send count lists the frontend can map.
    stats = {
        "size": s.size,
        "avg_cost": s.avg_cost,
        "total_pitch": s.total_pitch,
        "pitch": [{"pitch": p, "count": c} for p, c in s.pitch.items()],
        "cost_curve": [{"cost": cost, "count": c} for cost, c in s.cost_curve.items()],
        "types": [{"type": t, "count": c} for t, c in s.types.items()],
    }

    q = score_deck(pool)
    quality = {
        "overall": q.overall,
        "dimensions": [{"name": d.name, "score": d.score, "note": d.note} for d in q.dimensions],
    }

    return {
        "hero": _card_json(pool.hero),
        "format": pool.format,
        "legal": result.validation.ok,
        "rounds": result.rounds,
        "queries": result.queries,
        "finisher_notes": result.finisher_notes,
        "sideboard_notes": result.sideboard_notes,
        "warnings": result.warnings,
        "pool_total": len(pool.all_cards),
        "pool_max": get_format(pool.format).pool_max,
        "inventory": inventory,
        "deck": _grouped(pool.deck),
        "sideboard": sideboard,
        "stats": stats,
        "quality": quality,
        "explanation": explanation,
    }


# Static frontend last, so /api/* wins the route match.
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
