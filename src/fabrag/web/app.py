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
# Paging cap: ranking is a single matrix op over ~4.9k docs, so deep pages
# cost almost nothing server-side — the cap just bounds response size.
_MAX_DEPTH = 500


@app.get("/api/search")
def api_search(
    q: str,
    k: int = 12,
    offset: int = 0,
    source: str = "cards",
    color: str | None = None,
    pitch: int | None = None,
    card_class: str | None = None,
    legal: str | None = None,
    hero: str | None = None,
):
    """Ranked search with offset paging.

    The retriever has no native offset — it returns top-k. Paging is a slice
    of a deeper top-(offset+k), which re-ranks on every page request. That's
    deliberate simplicity: re-scoring the whole corpus is sub-millisecond, so
    caching page state server-side would be machinery without a payoff.
    """
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

    offset = max(0, offset)
    depth = min(offset + k, _MAX_DEPTH)
    results = rag.retrieve(
        q, k=depth, source=source, filters=filters, predicate=predicate
    )
    page = results[offset : offset + k]
    # More may exist if we filled the requested depth and haven't hit the cap.
    has_more = len(results) == depth and depth < _MAX_DEPTH
    return {
        "results": [_result_json(r) for r in page],
        "offset": offset,
        "has_more": has_more,
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

    try:
        result = build_deck(req.hero, req.strategy, req.format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    explanation = ""
    if req.explain:
        from ..generation import generate

        question = (
            f"This {result.deck.format} deck was built for {result.deck.hero.name} "
            f"with the strategy: {req.strategy!r}. Explain the game plan in 2-3 "
            f"short paragraphs, citing cards by name."
        )
        explanation = generate(question, deck_context(result))

    # Group copies for display: one entry per (name, pitch) with a count.
    from collections import Counter

    counts = Counter((c.name, c.pitch) for c in result.deck.cards)
    by_key = {(c.name, c.pitch): c for c in result.deck.cards}
    entries = []
    for (name, pitch), n in sorted(counts.items()):
        c = by_key[(name, pitch)]
        entry = _card_json(c)
        entry["copies"] = n
        entry["loadout"] = c.is_equipment or c.is_weapon
        entries.append(entry)

    return {
        "hero": _card_json(result.deck.hero),
        "format": result.deck.format,
        "legal": result.validation.ok,
        "rounds": result.rounds,
        "queries": result.queries,
        "finisher_notes": result.finisher_notes,
        "warnings": result.warnings,
        "cards": entries,
        "explanation": explanation,
    }


# Static frontend last, so /api/* wins the route match.
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
