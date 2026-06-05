"""Deck generation — Phase 6, the headline: constraint-guided generation.

Everything so far answers questions. This module *builds something*: a legal
decklist from a hero + a strategy, via the loop that is this phase's lesson:

    strategy --LLM--> retrieval queries --hybrid search--> candidate sheet
    candidates --LLM--> proposed decklist --validate_deck()--> errors
                            ^                                    |
                            +---- errors fed back, re-propose ---+

Why a loop at all? Because an LLM cannot be trusted to satisfy hard
constraints ("exactly 60 cards, max 3 copies, all hero-legal") in one shot —
counting and rule-keeping are exactly what 7B models are worst at. But it
doesn't need to be trusted: deck.py's validator is deterministic and explains
each failure in words ("4 copies of 'Sink Below'; max is 3"). Those error
strings go straight back into the next prompt. The LLM provides judgment
(which cards fit the strategy); the validator provides truth (whether the
list is legal); the loop converges where neither could alone. This is the
general shape of every agentic system: generator + verifier + feedback.

Two constraint tiers, deliberately distinct (mirroring rag.py's split):
  - LEGALITY is HARD: enforced by validate_deck and never waived.
  - GROUNDING is SOFT: the LLM is told to pick from the candidate sheet, and
    off-sheet picks are surfaced as warnings — but an off-sheet card that is
    hero-legal is accepted. (It came from the model's prior, not retrieval;
    we want to *see* that, not silently forbid a legal choice.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import ollama

from .cards import Card, expand_symbols, load_cards
from .deck import (
    MAX_COPIES,
    MIN_DECK_SIZE,
    Deck,
    DeckValidation,
    eligible_pool,
    find_hero,
    validate_deck,
)
from .generation import CHAT_MODEL
from .retrieval import Retriever, scope

# The candidate sheet + a 60-card JSON proposal won't fit Ollama's default
# context (often 4k, truncated SILENTLY — the model just never sees the tail
# of the prompt). qwen2.5 supports 32k; 16k fits comfortably in 12 GB VRAM.
NUM_CTX = 16384

MAX_ROUNDS = 4          # LLM repair attempts before the deterministic finisher
PER_QUERY_K = 20        # candidates retrieved per strategy-derived query


# =============================================================================
# Step 1 — strategy -> retrieval queries
# =============================================================================
_QUERY_PROMPT = """\
You translate a Flesh and Blood deck strategy into search queries. Given a hero
and a strategy, produce 4-6 short retrieval queries that would find cards
supporting that strategy. Cover different needs: core engine cards, attacks,
defensive options, resource/utility. Respond with JSON only:
{"queries": ["...", "..."]}\
"""

# If the LLM call fails or returns junk, deck building should still work —
# these generic needs apply to nearly every FAB deck.
_FALLBACK_QUERIES = [
    "efficient attack action cards",
    "strong defense reaction cards",
    "draw cards and generate resources",
    "cards with go again for tempo",
]


def derive_queries(hero: Card, strategy: str, *, model: str = CHAT_MODEL) -> list[str]:
    """Ask the LLM to decompose the strategy into retrieval queries.

    The decomposition step matters: one embedding of "aggro arcane wizard"
    averages all its facets into one vector (the same dilution problem chunking
    fights); four focused queries each retrieve their own slice of the pool.
    """
    hero_desc = (
        f"{hero.name} — classes: {', '.join(hero.classes) or 'none'}; "
        f"talents: {', '.join(hero.talents) or 'none'}"
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _QUERY_PROMPT},
                {"role": "user", "content": f"HERO: {hero_desc}\nSTRATEGY: {strategy}"},
            ],
            options={"temperature": 0.3},
            format="json",
        )
        queries = [str(q) for q in json.loads(resp.message.content)["queries"]]
        queries = [q.strip() for q in queries if q.strip()]
        if 2 <= len(queries) <= 8:
            return queries
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return _FALLBACK_QUERIES


# =============================================================================
# Step 2 — queries -> candidate sheet
# =============================================================================
def source_candidates(
    hero: Card,
    fmt: str,
    queries: list[str],
    retriever: Retriever,
    *,
    per_query: int = PER_QUERY_K,
) -> list[Card]:
    """Union of hero-pool hybrid search results across all strategy queries.

    Every candidate is GUARANTEED hero-legal (the predicate runs before
    ranking), so the proposer can't be led astray by its own reading list.
    Deduped by unique_id; pitch variants of one name stay distinct entries —
    picking the red or the blue version is a real deck-building decision.
    """
    from .deck import hero_pool_predicate  # local: avoids cycle at module load

    predicate = scope("cards", card_predicate=hero_pool_predicate(hero, fmt))
    seen: dict[str, Card] = {}
    for q in queries:
        for r in retriever.search(q, k=per_query, predicate=predicate):
            seen.setdefault(r.doc.unique_id, r.doc)
    return list(seen.values())


def _card_line(c: Card) -> str:
    """One candidate-sheet row: identity, the stats that matter, the effect."""
    bits = []
    if c.pitch is not None:
        bits.append(f"pitch {c.pitch}")
    if c.cost is not None:
        bits.append(f"cost {c.cost}")
    if c.power is not None:
        bits.append(f"power {c.power}")
    if c.defense is not None:
        bits.append(f"def {c.defense}")
    stats = ", ".join(bits) or "-"
    text = expand_symbols(c.text).replace("\n", " ")[:110]
    # The name is quoted so the model can see exactly what to copy into its
    # JSON — an earlier format without quotes had it copying whole lines as
    # "names", which resolved to nothing.
    return f'name: "{c.name}" | {stats} | {c.type_text} | {text}'


def candidate_sheet(candidates: list[Card]) -> str:
    return "\n".join(_card_line(c) for c in candidates)


# =============================================================================
# Step 3 — the propose / validate / repair loop
# =============================================================================
_PROPOSE_PROMPT = """\
You are an expert Flesh and Blood deck builder. Build a {fmt} main deck for the
given hero and strategy.

Rules you MUST follow:
1. Choose cards from the CANDIDATES list. Identify each card by its exact name
   and pitch value as shown.
2. The main deck must total AT LEAST {min_size} cards (sum of "copies").
3. At most 3 copies per card NAME — pitch variants of the same name share that
   limit (e.g. 2x red + 2x blue "Sink Below" = 4 copies = illegal).
4. Aim for a pitch balance roughly 60% pitch-1 (red), 10-20% pitch-2 (yellow),
   25-35% pitch-3 (blue), adjusted to the strategy.
5. Consistency beats variety: run 3 copies of your core cards rather than many
   singletons, and stay close to the {min_size}-card minimum.
6. Respond with JSON only:
{{"deck": [{{"name": "...", "pitch": 1, "copies": 3}}, ...]}}\
"""

_REPAIR_SUFFIX = """

YOUR PREVIOUS ATTEMPT WAS ILLEGAL. Fix ALL of these problems and return the
corrected complete deck (not a diff):
{errors}\
"""


@dataclass(frozen=True)
class Proposal:
    """One round's outcome: what the LLM picked and what the rules said."""

    cards: list[Card]               # resolved, every copy listed explicitly
    validation: DeckValidation
    unresolved: list[str]           # picks that matched no hero-legal card
    off_sheet: list[str]            # legal picks that weren't on the sheet


def _resolve_picks(
    picks: list[dict],
    candidates: list[Card],
    pool: list[Card],
) -> tuple[list[Card], list[str], list[str]]:
    """Map {"name", "pitch", "copies"} picks onto real Card objects.

    Resolution order: candidate sheet first, then the full hero-legal pool
    (off-sheet but legal -> warning, not error). A name that matches nothing
    hero-legal is unresolved — that's the anti-hallucination check, and it
    becomes a validation error fed back to the model.
    """
    by_key: dict[tuple[str, int | None], Card] = {}
    by_name: dict[str, Card] = {}
    sheet_ids = {c.unique_id for c in candidates}
    for c in [*candidates, *pool]:  # sheet entries win key collisions
        by_key.setdefault((c.name.lower(), c.pitch), c)
        by_name.setdefault(c.name.lower(), c)

    cards: list[Card] = []
    unresolved: list[str] = []
    off_sheet: list[str] = []
    for pick in picks:
        # Defensive parsing: models sometimes echo sheet decoration into the
        # name field ('Card Name [pitch 1, ...]' or quotes). Strip it rather
        # than failing the pick on cosmetics.
        name = str(pick.get("name", "")).strip().strip('"').split(" [")[0].split(" |")[0].strip()
        pitch = pick.get("pitch")
        pitch = int(pitch) if isinstance(pitch, (int, float)) else None
        copies = max(1, min(int(pick.get("copies", 1) or 1), 10))  # sanity clamp
        card = by_key.get((name.lower(), pitch)) or by_name.get(name.lower())
        if card is None:
            unresolved.append(name or "<blank>")
            continue
        if card.unique_id not in sheet_ids:
            off_sheet.append(card.name)
        cards.extend([card] * copies)
    return cards, unresolved, off_sheet


def propose_deck(
    hero: Card,
    strategy: str,
    fmt: str,
    candidates: list[Card],
    pool: list[Card],
    *,
    errors: list[str] | None = None,
    model: str = CHAT_MODEL,
) -> Proposal:
    """One LLM proposal round, resolved and validated."""
    min_size = MIN_DECK_SIZE.get(fmt, 60)
    system = _PROPOSE_PROMPT.format(fmt=fmt, min_size=min_size)
    user = (
        f"HERO: {hero.name} ({hero.type_text})\n"
        f"STRATEGY: {strategy}\n\n"
        f"CANDIDATES:\n{candidate_sheet(candidates)}"
    )
    if errors:
        user += _REPAIR_SUFFIX.format(errors="\n".join(f"- {e}" for e in errors))

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.4, "num_ctx": NUM_CTX},
        format="json",
    )
    try:
        picks = json.loads(resp.message.content).get("deck", [])
        if not isinstance(picks, list):
            picks = []
    except json.JSONDecodeError:
        picks = []

    cards, unresolved, off_sheet = _resolve_picks(picks, candidates, pool)
    validation = validate_deck(Deck(hero=hero, cards=cards, format=fmt))
    # Unresolved names are legality errors too: the model must re-pick them.
    if unresolved:
        validation = DeckValidation(
            ok=False,
            errors=validation.errors
            + [f"{n!r} is not a hero-legal card — pick from the candidates" for n in unresolved],
            warnings=validation.warnings,
        )
    return Proposal(cards, validation, unresolved, off_sheet)


# =============================================================================
# The deterministic finisher — legality is not negotiable
# =============================================================================
# Target main-deck pitch balance (fraction of cards by pitch value). The
# conventional FAB baseline: mostly red (your plays), a thin yellow band, and
# enough blue to pay costs. A heuristic, not a rule — used to GUIDE padding
# and to phrase quality warnings, never to fail a deck.
PITCH_TARGETS: dict[int, float] = {1: 0.55, 2: 0.12, 3: 0.33}


def _is_main_deck(card: Card) -> bool:
    return not (card.is_equipment or card.is_weapon)


def finish_deck(
    hero: Card,
    cards: list[Card],
    fmt: str,
    candidates: list[Card],
    pool: list[Card],
) -> tuple[list[Card], list[str]]:
    """Deterministically repair whatever the LLM left illegal.

    The repair loop usually converges; this is the backstop that makes
    `fabrag build` GUARANTEE a legal deck. Three mechanical passes, mirroring
    the three things validate_deck can complain about:

      1. DROP ineligible cards (wrong class/talent/format, non-deck types).
      2. CLAMP copies to the per-name limit (keep the first three seen).
      3. PAD up to the format minimum from the hero's pool — preferring the
         candidate sheet (strategy-relevant) and whichever pitch bucket is
         furthest under target, so the filler is playable, not random.

    Returns (repaired cards, notes describing what was done) — the notes go
    in the build report because silent repair would misrepresent the LLM's
    work as better than it was.
    """
    notes: list[str] = []
    min_size = MIN_DECK_SIZE.get(fmt, 60)

    # 1. drop ineligible
    from .deck import ineligibility_reasons

    kept: list[Card] = []
    dropped = 0
    for c in cards:
        if ineligibility_reasons(c, hero, fmt):
            dropped += 1
        else:
            kept.append(c)
    if dropped:
        notes.append(f"dropped {dropped} ineligible card(s)")

    # 2. clamp copies (by name — pitch variants share the budget)
    from collections import Counter

    counts: Counter[str] = Counter()
    clamped: list[Card] = []
    over = 0
    for c in kept:
        if counts[c.name] < MAX_COPIES:
            clamped.append(c)
            counts[c.name] += 1
        else:
            over += 1
    if over:
        notes.append(f"removed {over} over-limit cop(ies)")

    # 3. pad to the minimum, balance-aware
    main = [c for c in clamped if _is_main_deck(c)]
    deficit = min_size - len(main)
    if deficit > 0:
        # Fill order: sheet cards first (they matched the strategy), then the
        # rest of the pool; within each, sorted by name for reproducibility.
        fillers = sorted(
            (c for c in candidates if _is_main_deck(c)), key=lambda c: c.name
        ) + sorted(
            (c for c in pool if _is_main_deck(c)), key=lambda c: c.name
        )
        added = 0
        while added < deficit:
            # Which pitch bucket is furthest below target right now?
            n_main = len(main)
            gaps = {
                p: PITCH_TARGETS[p] - sum(1 for c in main if c.pitch == p) / max(n_main, 1)
                for p in PITCH_TARGETS
            }
            for want in sorted(gaps, key=gaps.get, reverse=True):
                pick = next(
                    (c for c in fillers
                     if c.pitch == want and counts[c.name] < MAX_COPIES),
                    None,
                )
                if pick is not None:
                    break
            else:
                pick = None
            if pick is None:  # no pitch-matching filler left — take anything legal
                pick = next((c for c in fillers if counts[c.name] < MAX_COPIES), None)
            if pick is None:
                notes.append("pool exhausted before reaching minimum deck size")
                break
            main.append(pick)
            clamped.append(pick)
            counts[pick.name] += 1
            added += 1
        if added:
            notes.append(f"padded {added} card(s) to reach {min_size}")

    return clamped, notes


def deck_warnings(cards: list[Card], fmt: str) -> list[str]:
    """Quality heuristics — never errors, always worth seeing."""
    out: list[str] = []
    main = [c for c in cards if _is_main_deck(c)]
    n = len(main)
    if not n:
        return out
    min_size = MIN_DECK_SIZE.get(fmt, 60)
    if n > min_size + 10:
        out.append(
            f"main deck is {n} cards; {min_size} is the minimum and fewer cards "
            f"means drawing your best ones more often"
        )
    for p, target in PITCH_TARGETS.items():
        actual = sum(1 for c in main if c.pitch == p) / n
        if abs(actual - target) > 0.20:
            out.append(
                f"pitch-{p} is {actual:.0%} of the main deck (typical ~{target:.0%})"
            )
    return out


# =============================================================================
# The build orchestrator
# =============================================================================
@dataclass(frozen=True)
class BuildResult:
    deck: Deck
    validation: DeckValidation       # of the FINAL deck (ok=True is the contract)
    rounds: int                      # LLM rounds used (1 = first try legal)
    queries: list[str]
    candidates: list[Card]
    off_sheet: list[str]
    finisher_notes: list[str] = field(default_factory=list)  # what repair did
    warnings: list[str] = field(default_factory=list)        # quality heuristics
    log: list[str] = field(default_factory=list)  # one summary line per round


def build_deck(
    hero_name: str,
    strategy: str = "a balanced, efficient deck",
    fmt: str = "cc",
    *,
    retriever: Retriever | None = None,
    max_rounds: int = MAX_ROUNDS,
    model: str = CHAT_MODEL,
) -> BuildResult:
    """The full pipeline: source candidates, loop propose -> validate, finish.

    The contract: the returned deck is LEGAL. The LLM loop gets `max_rounds`
    attempts to produce legality with judgment; whatever problems remain are
    mechanically repaired by finish_deck, and the repair is disclosed in
    finisher_notes. result.rounds + finisher_notes together tell you how much
    of the deck is judgment vs. backstop.
    """
    from .rag import get_retriever

    retriever = retriever if retriever is not None else get_retriever()
    cards = [d for d in retriever.docs if isinstance(d, Card)]
    hero = find_hero(hero_name, cards)
    pool = eligible_pool(cards, hero, fmt)

    queries = derive_queries(hero, strategy, model=model)
    candidates = source_candidates(hero, fmt, queries, retriever)

    min_size = MIN_DECK_SIZE.get(fmt, 60)

    def badness(p: Proposal) -> int:
        """How far from legal? Counting raw errors is a trap: an EMPTY deck
        scores exactly one error ("main deck has 0 cards") and would beat a
        78-card deck with four fixable problems — which is how an early
        version of this loop proudly returned nothing. The card deficit has
        to count per missing card."""
        deficit = max(0, min_size - len(p.cards))
        return len(p.validation.errors) + deficit

    log: list[str] = [f"pool {len(pool)} hero-legal cards; {len(candidates)} candidates from {len(queries)} queries"]
    best: Proposal | None = None
    errors: list[str] | None = None
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        prop = propose_deck(
            hero, strategy, fmt, candidates, pool, errors=errors, model=model
        )
        log.append(
            f"round {rounds}: {len(prop.cards)} cards, "
            f"{len(prop.validation.errors)} errors, {len(prop.unresolved)} unresolved"
        )
        if best is None or badness(prop) <= badness(best):
            best = prop
        if prop.validation.ok:
            break
        errors = prop.validation.errors

    # The backstop: mechanical repair of whatever the loop couldn't fix.
    final_cards = best.cards
    notes: list[str] = []
    if not best.validation.ok:
        final_cards, notes = finish_deck(hero, best.cards, fmt, candidates, pool)
        log.append(f"finisher: {'; '.join(notes) or 'no changes'}")

    deck = Deck(hero=hero, cards=final_cards, format=fmt)
    return BuildResult(
        deck=deck,
        validation=validate_deck(deck),
        rounds=rounds,
        queries=queries,
        candidates=candidates,
        off_sheet=best.off_sheet,
        finisher_notes=notes,
        warnings=deck_warnings(final_cards, fmt),
        log=log,
    )


# =============================================================================
# Explanation — the deck, justified from itself
# =============================================================================
def deck_context(result: BuildResult) -> str:
    """The decklist rendered as grounding context for the explanation call.

    We reuse generation.py's grounded-answer machinery: the decklist IS the
    context, so the rationale can only talk about cards actually in the deck —
    the same anti-hallucination contract as fabrag ask, pointed at our own
    output."""
    from collections import Counter

    groups: Counter[tuple[str, int | None]] = Counter(
        (c.name, c.pitch) for c in result.deck.cards
    )
    by_key = {(c.name, c.pitch): c for c in result.deck.cards}
    lines = []
    for (name, pitch), n in sorted(groups.items()):
        c = by_key[(name, pitch)]
        lines.append(f"{n}x {_card_line(c)}")
    return "\n".join(lines)


def explain_deck_stream(result: BuildResult, strategy: str):
    """Stream a short grounded rationale for the built deck."""
    from .generation import generate_stream

    question = (
        f"This {result.deck.format} deck was built for {result.deck.hero.name} "
        f"with the strategy: {strategy!r}. Explain the game plan in 2-3 short "
        f"paragraphs: how the deck executes the strategy, which cards are the "
        f"core engine, and what the pitch balance supports. Cite cards by name."
    )
    return generate_stream(question, deck_context(result))


if __name__ == "__main__":
    # Self-check: build a Blitz deck (40 cards — kinder to a 7B counting cards)
    result = build_deck("Ira, Crimson Haze", "fast aggressive ninja attacks", "blitz")
    print("\n".join(result.log))
    print(f"\nlegal: {result.validation.ok} after {result.rounds} round(s)")
    for e in result.validation.errors[:8]:
        print(f"  error: {e}")
    for w in result.warnings:
        print(f"  warning: {w}")
    if result.off_sheet:
        print(f"  off-sheet picks: {sorted(set(result.off_sheet))}")
    from collections import Counter
    names = Counter((c.name, c.pitch) for c in result.deck.cards)
    print(f"\n{len(result.deck.cards)} cards:")
    for (name, pitch), n in sorted(names.items()):
        print(f"  {n}x {name} (pitch {pitch})")
