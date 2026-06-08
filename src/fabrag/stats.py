"""Deck statistics — Phase 10, the numbers that describe a deck's shape.

build.py answers "is this deck legal?" and warns when the pitch balance looks
off. This module answers the descriptive question a player actually asks while
tuning: *what does my deck look like?* — the three views every FAB deck-builder
reads at a glance:

  - PITCH distribution: how many red / yellow / blue cards. Red cards are your
    plays, blue cards pay for them; the ratio is a deck's whole economy.
  - COST curve: the histogram of resource costs. A curve that piles up at high
    cost means hands you can't pay for.
  - TYPE breakdown: attacks vs. defense reactions vs. non-attack actions —
    roughly, how much of the deck is offense vs. answers.

Deliberately descriptive, not prescriptive: deck.py/build.py own the
"should it be different?" judgement (PITCH_TARGETS, warnings). Here we only
count, so the same numbers can feed the web stats panel, the CLI summary, and
(Phase 10's deck chat) an LLM reasoning about the deck.

Stats are computed over the STARTING DECK only — the inventory and sideboard
aren't part of the curve you draw from.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .cards import Card
from .pool import CardPool

# A card carries several category tokens ("Action" and "Attack" both sit on an
# attack action). For a single-bucket breakdown we pick the most specific, most
# play-relevant one in priority order.
_TYPE_PRIORITY: tuple[str, ...] = (
    "Attack Reaction", "Defense Reaction", "Instant", "Attack", "Block",
    "Action", "Equipment", "Weapon",
)


def primary_type(card: Card) -> str:
    """The single category that best describes how a card is played."""
    cats = set(card.categories)
    for t in _TYPE_PRIORITY:
        if t in cats:
            return t
    return card.categories[0] if card.categories else "Other"


@dataclass(frozen=True)
class DeckStats:
    """Descriptive shape of a deck's main cards. Counts only — percentages and
    bars are a presentation concern left to the caller."""

    size: int
    pitch: dict[int, int]        # pitch value (1/2/3) -> count; 0 = no pitch value
    cost_curve: dict[int, int]   # resource cost -> count; -1 = variable/none
    types: dict[str, int]        # primary card type -> count
    avg_cost: float | None       # mean cost over cards with a fixed cost
    total_pitch: int             # resources gained if every card were pitched


def deck_stats(pool: CardPool) -> DeckStats:
    """Compute the pitch / cost / type shape of `pool`'s starting deck."""
    deck = pool.deck
    pitch = Counter(c.pitch if c.pitch is not None else 0 for c in deck)
    # Cards with a variable/absent cost (X, free) bucket under -1 so they're
    # counted but never silently dropped from the total.
    cost_curve = Counter(c.cost if c.cost is not None else -1 for c in deck)
    types = Counter(primary_type(c) for c in deck)

    fixed_costs = [c.cost for c in deck if c.cost is not None]
    avg_cost = round(sum(fixed_costs) / len(fixed_costs), 2) if fixed_costs else None
    total_pitch = sum(c.pitch for c in deck if c.pitch is not None)

    return DeckStats(
        size=len(deck),
        pitch=dict(sorted(pitch.items())),
        cost_curve=dict(sorted(cost_curve.items())),
        types=dict(sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))),
        avg_cost=avg_cost,
        total_pitch=total_pitch,
    )


_PITCH_LABEL = {0: "no-pitch", 1: "red", 2: "yellow", 3: "blue"}


def render_stats(stats: DeckStats) -> str:
    """A compact text rendering for the CLI."""
    n = stats.size or 1
    lines = [f"Deck stats ({stats.size} cards, {stats.total_pitch} total pitch):"]

    pitch_bits = [
        f"{_PITCH_LABEL.get(p, p)} {c} ({c / n:.0%})" for p, c in stats.pitch.items()
    ]
    lines.append("  pitch: " + ", ".join(pitch_bits))

    avg = f"{stats.avg_cost}" if stats.avg_cost is not None else "n/a"
    curve_bits = [
        (f"X:{c}" if cost == -1 else f"{cost}:{c}") for cost, c in stats.cost_curve.items()
    ]
    lines.append(f"  cost (avg {avg}): " + " ".join(curve_bits))

    type_bits = [f"{t} {c}" for t, c in stats.types.items()]
    lines.append("  types: " + ", ".join(type_bits))
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-check: assemble a deck from a hero's pool and print its stats — no LLM.
    from .cards import load_cards
    from .deck import eligible_pool, find_hero

    cards = load_cards()
    hero = find_hero("Ira, Crimson Haze", cards)
    deck = [c for c in eligible_pool(cards, hero, "blitz")
            if not (c.is_equipment or c.is_weapon)][:40]
    pool = CardPool(hero=hero, deck=deck, inventory=[], sideboard=[], format="blitz")
    print(render_stats(deck_stats(pool)))
