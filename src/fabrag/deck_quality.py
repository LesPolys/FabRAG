"""Deck-quality scoring — carryover from Phase 6/10: legality is measured,
goodness wasn't.

build.py GUARANTEES a legal deck and warns on a couple of heuristics, but
"legal" and "good" are different questions — an empty-of-defense, top-heavy
40 cards can be perfectly legal and unplayable. This module scores the things
that make a constructed deck actually *work*, as an interpretable scorecard.

Honesty about what this is: there's no ground-truth "deck rating" to regress
against (unlike the retrieval gold set), so this is NOT a learned or validated
metric — it's a panel of defensible heuristics with explicit targets, each
scored 0..1 with a human-readable note. Its job is to be *comparable and
legible*: to tell you how two builds differ and why, and to give the deck chat
something concrete to reason about. Treat the numbers as a dashboard, not a
verdict.

Dimensions (the overall is their mean):
  - PITCH BALANCE   — closeness to the conventional red/yellow/blue split.
  - CURVE HEALTH    — are most cards cheap enough to actually play?
  - ROLE COVERAGE   — offense AND defensive answers AND utility/advantage.
  - CONSISTENCY     — (copy-limit formats only) core cards run at the limit,
                      so you draw them reliably. Omitted for Blitz, where the
                      1-copy rule makes every card a singleton by definition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .formats import get_format
from .pool import CardPool
from .stats import primary_type

# The conventional FAB pitch split (mirrors build.PITCH_TARGETS; duplicated as a
# scoring target so this module stays a pure, dependency-light consumer).
PITCH_TARGETS: dict[int, float] = {1: 0.55, 2: 0.12, 3: 0.33}

# Primary types that count as each role.
_OFFENSE = {"Attack"}
_DEFENSE = {"Defense Reaction", "Block"}
_UTILITY = {"Action", "Instant", "Attack Reaction", "Item", "Aura", "Resource"}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class QualityDimension:
    name: str
    score: float   # 0..1
    note: str


@dataclass(frozen=True)
class DeckQuality:
    overall: float
    dimensions: list[QualityDimension]


def _pitch_dimension(deck) -> QualityDimension:
    n = len(deck) or 1
    actual = {p: sum(1 for c in deck if c.pitch == p) / n for p in PITCH_TARGETS}
    dev = sum(abs(actual[p] - PITCH_TARGETS[p]) for p in PITCH_TARGETS) / len(PITCH_TARGETS)
    # Mean absolute deviation of ~0.33 (wildly off) floors the score at 0.
    score = _clamp(1 - dev / 0.33)
    worst = max(PITCH_TARGETS, key=lambda p: abs(actual[p] - PITCH_TARGETS[p]))
    note = (f"red {actual[1]:.0%}/yellow {actual[2]:.0%}/blue {actual[3]:.0%} "
            f"(targets 55/12/33; furthest off: pitch-{worst})")
    return QualityDimension("pitch balance", round(score, 2), note)


def _curve_dimension(deck) -> QualityDimension:
    n = len(deck) or 1
    # Cards you can pay for early. Variable/None cost (X) counts as playable —
    # it's usually scaled to available resources, not a fixed high cost.
    affordable = sum(1 for c in deck if c.cost is None or c.cost <= 1)
    share = affordable / n
    fixed = [c.cost for c in deck if c.cost is not None]
    avg = sum(fixed) / len(fixed) if fixed else 0.0
    high = sum(1 for c in deck if c.cost is not None and c.cost >= 3)
    note = f"{share:.0%} cost <=1, avg cost {avg:.2f}, {high} at cost >=3"
    return QualityDimension("curve health", round(_clamp(share), 2), note)


def _role_dimension(deck) -> QualityDimension:
    n = len(deck) or 1
    types = Counter(primary_type(c) for c in deck)
    offense = sum(types[t] for t in _OFFENSE)
    defense = sum(types[t] for t in _DEFENSE)
    utility = sum(types[t] for t in _UTILITY)
    # Soft targets relative to deck size: a real attack base, a few answers,
    # a few advantage/utility cards. Each capped at 1.0.
    off_s = _clamp(offense / (0.40 * n))
    def_s = _clamp(defense / max(2, 0.08 * n))
    utl_s = _clamp(utility / max(3, 0.10 * n))
    score = (off_s + def_s + utl_s) / 3
    note = f"offense {offense}, defense {defense}, utility {utility} (of {n})"
    return QualityDimension("role coverage", round(score, 2), note)


def _consistency_dimension(deck, max_copies: int) -> QualityDimension:
    n = len(deck) or 1
    distinct = len({(c.name, c.pitch) for c in deck})
    ratio = distinct / n                       # 1.0 = all singletons
    best = 1 / max_copies                       # all run at the limit
    score = _clamp((1 - ratio) / (1 - best)) if best < 1 else 1.0
    at_limit = sum(1 for _, k in Counter((c.name, c.pitch) for c in deck).items() if k == max_copies)
    note = f"{distinct} distinct cards, {at_limit} run at the {max_copies}-copy limit"
    return QualityDimension("consistency", round(score, 2), note)


def score_deck(pool: CardPool) -> DeckQuality:
    """Score `pool`'s starting deck across the quality dimensions."""
    deck = pool.deck
    rules = get_format(pool.format)
    dims = [_pitch_dimension(deck), _curve_dimension(deck), _role_dimension(deck)]
    if rules.max_copies > 1:  # consistency-by-copies is meaningless in 1-copy formats
        dims.append(_consistency_dimension(deck, rules.max_copies))
    overall = round(sum(d.score for d in dims) / len(dims), 2) if dims else 0.0
    return DeckQuality(overall, dims)


def render_quality(q: DeckQuality) -> str:
    """Compact text scorecard for the CLI."""
    lines = [f"Deck quality: {q.overall:.2f} / 1.00"]
    for d in q.dimensions:
        lines.append(f"  {d.name:14} {d.score:.2f}  - {d.note}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-check: score a synthetic deck (no LLM).
    from .cards import load_cards
    from .deck import eligible_pool, find_hero

    cards = load_cards()
    hero = find_hero("Dorinthea Ironsong", cards)
    deck = [c for c in eligible_pool(cards, hero, "cc")
            if not (c.is_equipment or c.is_weapon)][:60]
    pool = CardPool(hero=hero, deck=deck, inventory=[], sideboard=[], format="cc")
    print(render_quality(score_deck(pool)))
