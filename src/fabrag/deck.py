"""Deck-construction rules — Phase 3, the FAB legality layer.

retrieval.py answers "is this card *about* X?" and CardFilter answers "does it
have property Y?". Neither knows the game's deck-building rules. This module
adds them, in two tiers:

  1. ELIGIBILITY — given a hero, which cards may legally go in the deck?
     A non-Generic card must match the hero's CLASS, and any card with a
     TALENT must match one of the hero's talents (an Ice Wizard may run Ice
     cards but not Lightning ones), plus the card must be legal in the format.

  2. VALIDATION — is a finished decklist legal? Min deck size, the max-3-copies
     rule, every card eligible for the hero.

The key formal idea is the SUBSET rule: a card is class/talent-legal when its
classes and talents are each a *subset* of the hero's. That's why eligibility
can't be expressed with CardFilter's any-of matching — a card with talents
{Ice, Shadow} shares Ice with an Ice/Lightning hero but is still illegal
(Shadow isn't in the hero's talents). So we model it as its own predicate and
feed it to retrieval via the search `predicate` hook.

Known simplifications (room to grow): 'Legendary' single-copy cards aren't
flagged in the source data so we apply the format's flat copy limit. Equipment
slot limits and the whole card-pool registration moved up into pool.py (Phase 9),
which is where Blitz's exact-40 and pool-cap rules live; this module stays about
a hero's eligible pool and a single deck's legality.

Per-format rules (sizes, copy limits, hero age) come from formats.py — one
table, read by every layer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .cards import Card
from .formats import get_format, normalize_format

# CR 1.3.2 splits cards into hero-, token-, deck-, and arena-cards; only
# deck-cards may start in a deck. These type keywords mark the non-deck kinds
# present in the data (each has ZERO cards with a pitch value — the deck-card
# hallmark). Found the hard way: Phase 6's generator legally "drafted" the
# Quicken token before this exclusion existed. Figments, Constructs,
# Invocations and Afflictions DO pitch and stay deckable (e.g. Dromai's
# Invocations are real main-deck cards).
NON_DECK_TYPES: frozenset[str] = frozenset({"Token", "Macro", "Landmark", "Demi-Hero"})


# --- hero lookup -------------------------------------------------------------
def find_hero(name: str, cards: Iterable[Card]) -> Card:
    """Resolve a hero by (case-insensitive) name; exact match wins over partial.

    Raises ValueError if nothing matches or a partial query is ambiguous.
    """
    heroes = [c for c in cards if c.is_hero]
    low = name.strip().lower()
    exact = [h for h in heroes if h.name.lower() == low]
    if exact:
        return exact[0]
    partial = [h for h in heroes if low in h.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"No hero matching {name!r}.")
    raise ValueError(
        f"Ambiguous hero {name!r}; matches: {sorted(h.name for h in partial)}"
    )


# --- tier 1: eligibility -----------------------------------------------------
def ineligibility_reasons(card: Card, hero: Card, fmt: str = "cc") -> list[str]:
    """Why `card` may NOT go in `hero`'s `fmt` deck — empty list means eligible."""
    if not hero.is_hero:
        raise ValueError(f"{hero.name!r} is not a hero card.")

    reasons: list[str] = []
    non_deck = NON_DECK_TYPES.intersection(card.types)
    if non_deck:
        reasons.append(f"{sorted(non_deck)} cards can't start in a deck (CR 1.3.2)")
    if not card.legality.is_legal(fmt):
        reasons.append(f"not legal in {normalize_format(fmt)}")

    off_class = set(card.classes) - set(hero.classes)
    if off_class:
        have = ", ".join(hero.classes) or "Generic-only"
        reasons.append(f"class {sorted(off_class)} not in hero's ({have})")

    off_talent = set(card.talents) - set(hero.talents)
    if off_talent:
        have = ", ".join(hero.talents) or "no talents"
        reasons.append(f"talent {sorted(off_talent)} not in hero's ({have})")
    return reasons


def is_eligible(card: Card, hero: Card, fmt: str = "cc") -> bool:
    """True if `card` may legally be included in `hero`'s `fmt` deck."""
    return not ineligibility_reasons(card, hero, fmt)


def hero_age_error(hero: Card, fmt: str) -> str | None:
    """Why `hero` is the wrong age for `fmt`, or None if fine.

    A hero-vs-format check (not a card-eligibility one): CC wants an adult hero,
    Blitz a young one. Formats with `hero_age=None` (e.g. commoner here) impose
    no restriction.
    """
    age = get_format(fmt).hero_age
    if age == "young" and not hero.is_young:
        return f"{hero.name} is an adult hero; {normalize_format(fmt)} requires a young hero"
    if age == "adult" and hero.is_young:
        return f"{hero.name} is a young hero; {normalize_format(fmt)} requires an adult hero"
    return None


def copy_limit_errors(cards: Iterable[Card], max_copies: int) -> list[str]:
    """Copy-limit violations counted per UNIQUE card — (name, pitch), per
    CR 2.7.1/2.8.1, NOT per name. So 3 red + 3 blue of one name is two unique
    cards (legal at CC's limit of 3), while 4 of a single pitch is not."""
    errors: list[str] = []
    for (name, pitch), n in Counter((c.name, c.pitch) for c in cards).items():
        if n > max_copies:
            where = f" (pitch {pitch})" if pitch is not None else ""
            errors.append(f"{n} copies of {name!r}{where}; max is {max_copies}")
    return errors


def hero_pool_predicate(hero: Card, fmt: str = "cc") -> Callable[[Card], bool]:
    """A row predicate for Retriever.search: keep only the hero's legal pool."""
    hero_classes = set(hero.classes)
    hero_talents = set(hero.talents)

    def keep(card: Card) -> bool:
        if card.is_hero or NON_DECK_TYPES.intersection(card.types):
            return False  # heroes/tokens/etc. can't start in a deck
        return (
            card.legality.is_legal(fmt)
            and set(card.classes) <= hero_classes
            and set(card.talents) <= hero_talents
        )

    return keep


def eligible_pool(cards: Iterable[Card], hero: Card, fmt: str = "cc") -> list[Card]:
    """Every card from `cards` that `hero` may legally include."""
    keep = hero_pool_predicate(hero, fmt)
    return [c for c in cards if keep(c)]


# --- tier 2: deck validation -------------------------------------------------
@dataclass(frozen=True)
class Deck:
    """A hero plus the non-hero cards chosen for the deck (main deck + loadout).

    `cards` lists every copy explicitly (three copies of a card appear thrice),
    which is what the copy-limit check counts.
    """

    hero: Card
    cards: list[Card] = field(default_factory=list)
    format: str = "cc"


@dataclass(frozen=True)
class DeckValidation:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _is_loadout(card: Card) -> bool:
    """Equipment and weapons form the loadout — outside the main-deck count."""
    return card.is_equipment or card.is_weapon


def validate_deck(deck: Deck) -> DeckValidation:
    """Check a decklist against the format's construction rules."""
    errors: list[str] = []
    warnings: list[str] = []
    rules = get_format(deck.format)
    fmt = rules.key

    if not deck.hero.is_hero:
        errors.append(f"{deck.hero.name!r} is not a hero card.")
        return DeckValidation(False, errors, warnings)  # rest assumes a real hero

    age_problem = hero_age_error(deck.hero, fmt)
    if age_problem:
        errors.append(age_problem)

    # Every card must be eligible for the hero.
    for card in deck.cards:
        reasons = ineligibility_reasons(card, deck.hero, fmt)
        if reasons:
            errors.append(f"{card.name}: " + "; ".join(reasons))

    # Main-deck size (loadout excluded). deck_max == deck_min means the format
    # is exact-size (Blitz/Commoner are exactly 40, not "at least 40").
    main = [c for c in deck.cards if not _is_loadout(c)]
    n = len(main)
    if rules.deck_max == rules.deck_min:
        if n != rules.deck_min:
            errors.append(f"main deck has {n} cards; {fmt} requires exactly {rules.deck_min}")
    elif n < rules.deck_min:
        errors.append(f"main deck has {n} cards; {fmt} requires at least {rules.deck_min}")

    # Copy limit, per unique card.
    errors.extend(copy_limit_errors(deck.cards, rules.max_copies))

    return DeckValidation(not errors, errors, warnings)
