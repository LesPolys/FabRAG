"""Card-pool registration — Phase 9, modelling deck-building as the game does.

deck.py validates a *deck* — a hero plus a flat pile of cards. But you don't
register a deck; you register a CARD-POOL (CR 1.3.2, TRP §7): a hero, a starting
**deck**, the **arena cards** you start equipped with (one per slot), and a
**sideboard** of extra registered cards you may swap to between games. This
module adds that structure and the rules that only make sense once it exists:

  - INVENTORY slots: at most one arena-card per Head/Chest/Arms/Legs/Off-Hand,
    and weapons limited by hand count (CR 4.1.4a). A pool may *register* many
    chest pieces; only one starts equipped.
  - POOL CAP: deck + arena cards combined is capped per format (CC 80, Blitz 52).
  - Copy + size + hero-age rules, per format, reusing deck.py + formats.py.

The split mirrors the rest of the codebase: deck.py owns "is this one deck
legal?", this module owns "is this whole registration legal?". validate_pool
composes deck.py's primitives (ineligibility_reasons, copy_limit_errors,
hero_age_error) rather than re-deriving them — one source of truth per rule.

Known simplification: hand count assumes the common 2-hand hero (a 2H weapon
fills both hands, a 1H weapon one); heroes/effects that change available hands
aren't modelled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .cards import Card
from .deck import (
    NON_DECK_TYPES,
    copy_limit_errors,
    hero_age_error,
    ineligibility_reasons,
)
from .formats import get_format

MAX_HANDS = 2  # weapon zones on the common hero; a 2H weapon uses both


def weapon_hands(card: Card) -> int:
    """Hands a weapon occupies: 2 for a two-handed weapon, else 1."""
    return 2 if "2H" in card.types else 1


@dataclass(frozen=True)
class CardPool:
    """A full registration: hero + starting deck + equipped inventory + sideboard.

    Every list holds each copy explicitly (like Deck.cards), so the copy and
    pool-size counts are just len(). `deck` holds deck-cards only; `inventory`
    holds the arena-cards (weapons/equipment) that START equipped; `sideboard`
    holds everything else registered (extra equipment, swap options).
    """

    hero: Card
    deck: list[Card] = field(default_factory=list)
    inventory: list[Card] = field(default_factory=list)
    sideboard: list[Card] = field(default_factory=list)
    format: str = "cc"

    @property
    def all_cards(self) -> list[Card]:
        """Every registered card — what the pool cap and copy limits count."""
        return [*self.deck, *self.inventory, *self.sideboard]


@dataclass(frozen=True)
class PoolValidation:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _arena(card: Card) -> bool:
    return card.is_equipment or card.is_weapon


def _inventory_slot_errors(inventory: list[Card]) -> list[str]:
    """At most one arena-card per body slot, and weapons within the hand limit."""
    errors: list[str] = []

    slots = Counter(c.equipment_slot for c in inventory if c.equipment_slot)
    for slot, n in slots.items():
        if n > 1:
            errors.append(f"{n} {slot} items equipped; only 1 fits the {slot} slot")

    hands = sum(weapon_hands(c) for c in inventory if c.is_weapon)
    if hands > MAX_HANDS:
        errors.append(f"weapons need {hands} hands; only {MAX_HANDS} available")

    return errors


def validate_pool(pool: CardPool) -> PoolValidation:
    """Check a full card-pool registration against its format's rules."""
    errors: list[str] = []
    warnings: list[str] = []

    if not pool.hero.is_hero:
        return PoolValidation(False, [f"{pool.hero.name!r} is not a hero card."], warnings)

    rules = get_format(pool.format)
    fmt = rules.key

    age_problem = hero_age_error(pool.hero, fmt)
    if age_problem:
        errors.append(age_problem)

    # Structural: deck holds deck-cards, inventory holds arena-cards.
    for c in pool.deck:
        if _arena(c):
            errors.append(f"{c.name} is equipment/a weapon — it belongs in the inventory")
    for c in pool.inventory:
        if not _arena(c):
            errors.append(f"{c.name} is a deck-card — it can't be equipped as inventory")

    # Eligibility across the whole registration (class/talent/legality/non-deck).
    for c in pool.all_cards:
        reasons = ineligibility_reasons(c, pool.hero, fmt)
        if reasons:
            errors.append(f"{c.name}: " + "; ".join(reasons))

    # Deck size — exact when deck_max == deck_min, else a minimum.
    n_deck = len(pool.deck)
    if rules.deck_max == rules.deck_min:
        if n_deck != rules.deck_min:
            errors.append(f"deck has {n_deck} cards; {fmt} requires exactly {rules.deck_min}")
    elif n_deck < rules.deck_min:
        errors.append(f"deck has {n_deck} cards; {fmt} requires at least {rules.deck_min}")

    # Inventory slots + hands.
    errors.extend(_inventory_slot_errors(pool.inventory))

    # Copy limit across the WHOLE pool (deck + inventory + sideboard).
    errors.extend(copy_limit_errors(pool.all_cards, rules.max_copies))

    # Pool cap: deck + arena cards combined.
    total = len(pool.all_cards)
    if rules.pool_max is not None and total > rules.pool_max:
        errors.append(
            f"card-pool has {total} cards; {fmt} allows at most {rules.pool_max}"
        )

    # Warnings — never failures, always worth seeing.
    if not any(c.is_weapon for c in pool.inventory):
        warnings.append("no weapon equipped — most heroes want one to attack with")

    return PoolValidation(not errors, errors, warnings)


if __name__ == "__main__":
    # Self-check: assemble a real Blitz pool from the eligible pool, then mutate
    # it to demonstrate each class of failure. (No LLM — pure rules.)
    from .cards import load_cards
    from .deck import eligible_pool, find_hero

    cards = load_cards()
    young = next(c for c in cards if c.is_hero and c.is_young)
    pool_cards = eligible_pool(cards, young, "blitz")
    deck_cards = [c for c in pool_cards if not (c.is_equipment or c.is_weapon)][:40]
    weapon = next((c for c in pool_cards if c.is_weapon), None)
    head = next((c for c in pool_cards if c.equipment_slot == "Head"), None)
    inventory = [c for c in (weapon, head) if c is not None]

    legal = CardPool(young, deck_cards, inventory, [], "blitz")
    v = validate_pool(legal)
    print(f"Blitz pool for {young.name}: {len(deck_cards)} deck + {len(inventory)} inventory")
    print(f"  legal={v.ok}  errors={v.errors}  warnings={v.warnings}\n")

    # Deliberately illegal: one card too few, a duplicate (Blitz max 1), and an
    # adult hero in Blitz.
    adult = next(c for c in cards if c.is_hero and not c.is_young)
    bad = CardPool(adult, deck_cards[:-1] + [deck_cards[0]], inventory, [], "blitz")
    v = validate_pool(bad)
    print(f"Deliberately-illegal pool (adult hero, dup card): legal={v.ok}")
    for e in v.errors:
        print(f"  error: {e}")
