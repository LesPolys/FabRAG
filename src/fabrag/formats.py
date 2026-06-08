"""Per-format construction rules — Phase 9, one source of truth.

deck.py knew a flat "min 60, max 3 copies" because Phase 3 only needed Classic
Constructed. Phase 9 reads the official Tournament Rules & Policy
(rules.fabtcg.com/en/trp/07-constructed-formats/) and finds each format is its
own ruleset:

  - Classic Constructed: an ADULT hero, a deck of AT LEAST 60 cards, up to 3
    copies of each unique card, and a card-pool capped at 80 (deck + arena).
  - Blitz: a YOUNG hero, EXACTLY 40 cards, up to 1 copy of each unique card,
    pool capped at 52.

The numbers were scattered (MIN_DECK_SIZE here, MAX_COPIES there, and the
format-alias maps duplicated in deck.py and cards.py). This module collects
them into a `FormatRules` record so every layer — eligibility, deck validation,
pool validation, the build prompt — reads the *same* rules, and adding a format
is one table row instead of a grep across the codebase.

A subtlety worth its own note: "up to N copies of each UNIQUE card" counts by
(name, pitch), not by name — CR 2.7.1/2.8.1: "names and pitch values are used
to determine uniqueness of a card". So 3 red + 3 blue copies of one name is six
cards but two unique cards, each within the CC limit of 3. Earlier code counted
by name and got this wrong; the limit lives here now so there's one place to be
right.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormatRules:
    """The construction rules for one constructed format.

    `deck_max == deck_min` encodes an exact-size format (Blitz/Commoner are
    exactly 40); `deck_max is None` means "no upper bound on the deck itself"
    (CC, where the 80-card *pool* cap is the real ceiling). `hero_age is None`
    means the format puts no age restriction on the hero. `max_copies` is per
    UNIQUE card — (name, pitch) — not per name.
    """

    key: str                 # canonical format key ("cc", "blitz", ...)
    label: str               # human-readable name
    hero_age: str | None     # "young" | "adult" | None (no restriction)
    deck_min: int            # minimum starting-deck size
    deck_max: int | None     # exact size if == deck_min; None if unbounded
    pool_max: int | None     # combined deck + arena-card cap; None if unbounded
    max_copies: int          # max copies of each unique (name, pitch) card


# The table. cc and blitz are authored from the TRP; commoner keeps the size it
# had in Phase 3 but its finer points (Common-rarity-only, hero age) are left
# unmodeled — a documented simplification, not a claim of correctness.
FORMATS: dict[str, FormatRules] = {
    "cc": FormatRules("cc", "Classic Constructed", "adult", 60, None, 80, 3),
    "blitz": FormatRules("blitz", "Blitz", "young", 40, 40, 52, 1),
    "commoner": FormatRules("commoner", "Commoner", None, 40, 40, None, 3),
}

# Friendly spellings -> canonical keys. Mirrors (and now centralizes) the alias
# maps that were duplicated in deck.py and cards.py:Legality.
_ALIASES: dict[str, str] = {
    "classic_constructed": "cc", "classic": "cc",
    "living_legend": "ll", "silverage": "silver_age",
    "ultimate_pit_fight": "upf", "pit_fight": "upf",
}


def normalize_format(fmt: str) -> str:
    """Canonicalize a format name: lowercase, spaces/hyphens to underscores,
    then resolve aliases ('Classic Constructed' -> 'cc')."""
    key = fmt.strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(key, key)


def get_format(fmt: str) -> FormatRules:
    """Resolve a (possibly aliased) format name to its FormatRules.

    Raises ValueError for a format we have no construction rules for — note
    this is narrower than Legality.is_legal, which knows six formats; we only
    build decks for the constructed formats in FORMATS.
    """
    key = normalize_format(fmt)
    rules = FORMATS.get(key)
    if rules is None:
        raise ValueError(
            f"No deck-construction rules for format {fmt!r}; "
            f"known: {sorted(FORMATS)}"
        )
    return rules
