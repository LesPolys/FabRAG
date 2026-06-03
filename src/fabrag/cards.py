"""The Card data model — Phase 1, the data layer.

Goal: turn the raw, *stringly-typed* records in data/raw/card.json into clean,
validated `Card` objects, and make the central architectural split explicit:

  - STRUCTURED metadata  (color, pitch, stats, type facets, legality)  -> FILTERING
  - free TEXT            (name + type line + keywords + rules text)    -> EMBEDDING

That split is the seed of the *hybrid retrieval* we build in Phase 4: exact-match
filters answer "is this a legal blue Wizard card?", embeddings answer "is this card
*about* dealing arcane damage?". Neither alone is enough; production RAG uses both.
"""

from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

# data/raw/card.json, resolved relative to this file (src/fabrag/cards.py -> repo root).
RAW_CARD_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "card.json"

# --- Domain vocabulary -------------------------------------------------------
# The raw `types` list flattens several ORTHOGONAL dimensions into one array,
# e.g. ["Wizard", "Weapon", "Staff", "2H"] = class + category + two subtypes.
# The dataset doesn't categorize these tokens, so we encode the taxonomy here.
# These sets are curated domain heuristics — easy to extend, and lossless: any
# token we don't recognize still lives on in `Card.types` and `Card.other_types`.

CLASSES: frozenset[str] = frozenset({
    "Adjudicator", "Assassin", "Bard", "Brute", "Guardian", "Illusionist",
    "Mechanologist", "Mercenary", "Merchant", "Necromancer", "Ninja",
    "Pit-Fighter", "Ranger", "Runeblade", "Shapeshifter", "Thief", "Warrior",
    "Wizard",
})  # note: "Generic" means class-agnostic, handled separately below.

TALENTS: frozenset[str] = frozenset({
    "Chaos", "Draconic", "Earth", "Elemental", "Ice", "Light", "Lightning",
    "Mystic", "Royal", "Shadow",
})

# "What kind of card is this" — the supertypes we filter on.
CARD_CATEGORIES: frozenset[str] = frozenset({
    "Action", "Instant", "Attack", "Attack Reaction", "Defense Reaction",
    "Block", "Equipment", "Weapon", "Hero", "Demi-Hero", "Token", "Resource",
    "Item", "Aura", "Landmark", "Trap", "Ally", "Mentor", "Figment",
    "Invocation", "Evo", "Base", "Event", "Affliction", "Companion",
    "Construct", "Macro",
})


# FAB card text uses curly-brace symbols (e.g. "Gain 3{h}", "+4{p}") that are
# opaque to an embedding model. Expanding them to real words is essential for
# semantic search — without it, "Gain 3{h}" does NOT land near "gain life".
# Meanings inferred from the data; leading spaces keep "3{h}" -> "3 life".
SYMBOLS: dict[str, str] = {
    "{p}": " power",
    "{r}": " resource",
    "{d}": " defense",
    "{h}": " life",
    "{i}": " intellect",
    "{t}": " tap",
    "{u}": " untap",
    "{c}": " chi",
}


def expand_symbols(text: str) -> str:
    """Replace FAB's curly-brace game symbols with words, for embedding."""
    text = text.replace("{r]", "{r}")  # repair a malformed token in the source data
    for sym, word in SYMBOLS.items():
        text = text.replace(sym, word)
    return re.sub(r"[ \t]{2,}", " ", text)  # collapse the spaces we introduced


def parse_stat(raw: str | None) -> int | None:
    """FAB stats arrive as strings: '' (none), digits, or variable markers.

    Examples seen in the data: '6', '', 'X', 'XX', 'X1', '*'.
    Returns an int only for a *fixed* numeric value; '' and variable markers
    (X / *) return None — the card's rules text carries the variable meaning.
    A naive int(raw) would crash on ~50 cards.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    if raw.lstrip("-").isdigit():
        return int(raw)
    return None  # variable (X, *, XX, ...) -> not a fixed number


class Legality(BaseModel):
    """Per-format playability, straight from card.json's booleans.

    All six formats the dataset tracks are stored losslessly. A card is
    *effectively* legal in a format when it's legal AND not banned/suspended/
    retired — but the exact rule differs per format because the formats expose
    different flags:
      - CC / Blitz : legal & not (banned | suspended | living_legend)
      - Commoner   : legal & not (banned | suspended)
      - Living Legend : legal & not banned   (`restricted` = max 1 copy, still legal)
      - Silver Age : legal & not banned
      - UPF        : not banned   (no per-card `legal` flag exists; ban-list only)
    """

    # Classic Constructed
    cc_legal: bool
    cc_banned: bool
    cc_suspended: bool
    cc_living_legend: bool
    # Blitz
    blitz_legal: bool
    blitz_banned: bool
    blitz_suspended: bool
    blitz_living_legend: bool
    # Commoner
    commoner_legal: bool
    commoner_banned: bool
    commoner_suspended: bool
    # Living Legend
    ll_legal: bool
    ll_banned: bool
    ll_restricted: bool
    # Silver Age
    silver_age_legal: bool
    silver_age_banned: bool
    # Ultimate Pit Fight (multiplayer) — ban-list only, no per-card legal flag
    upf_banned: bool

    @property
    def is_cc_legal(self) -> bool:
        """Effective Classic Constructed legality (the flagship format)."""
        return self.cc_legal and not (self.cc_banned or self.cc_suspended or self.cc_living_legend)

    @property
    def is_blitz_legal(self) -> bool:
        return self.blitz_legal and not (
            self.blitz_banned or self.blitz_suspended or self.blitz_living_legend
        )

    @property
    def is_commoner_legal(self) -> bool:
        return self.commoner_legal and not (self.commoner_banned or self.commoner_suspended)

    @property
    def is_ll_legal(self) -> bool:
        # `restricted` limits a card to 1 copy but does NOT make it illegal.
        return self.ll_legal and not self.ll_banned

    @property
    def is_silver_age_legal(self) -> bool:
        return self.silver_age_legal and not self.silver_age_banned

    @property
    def is_upf_legal(self) -> bool:
        return not self.upf_banned

    @property
    def by_format(self) -> dict[str, bool]:
        """Effective legality keyed by canonical format name."""
        return {
            "cc": self.is_cc_legal,
            "blitz": self.is_blitz_legal,
            "commoner": self.is_commoner_legal,
            "ll": self.is_ll_legal,
            "silver_age": self.is_silver_age_legal,
            "upf": self.is_upf_legal,
        }

    # Friendly aliases -> canonical keys used in `by_format`.
    _FORMAT_ALIASES: ClassVar[dict[str, str]] = {
        "classic_constructed": "cc",
        "classic": "cc",
        "living_legend": "ll",
        "silverage": "silver_age",
        "ultimate_pit_fight": "upf",
        "pit_fight": "upf",
    }

    def is_legal(self, fmt: str) -> bool:
        """Generic accessor, e.g. is_legal('cc'), is_legal('Living Legend')."""
        key = fmt.strip().lower().replace(" ", "_").replace("-", "_")
        key = self._FORMAT_ALIASES.get(key, key)
        legal = self.by_format
        if key not in legal:
            raise ValueError(f"Unknown format {fmt!r}; valid: {sorted(legal)}")
        return legal[key]


class Card(BaseModel):
    """A single unique Flesh and Blood card, cleaned and typed."""

    # --- identity ---
    unique_id: str
    name: str

    # --- resource system (structured -> filtering) ---
    color: str | None          # pitch color: Red / Yellow / Blue (None for non-pitch cards)
    pitch: int | None          # pitch value 1 / 2 / 3
    cost: int | None           # resource cost (None if free or variable 'X')
    power: int | None          # attack power (None if not an attack or variable '*')
    defense: int | None        # defense value
    health: int | None         # hero health
    intelligence: int | None   # hero intelligence (cards drawn)
    arcane: int | None         # arcane damage

    # --- types: lossless raw list + derived facets ---
    types: list[str]
    traits: list[str]
    keywords: list[str]        # from raw `card_keywords`, e.g. "Go Again", "Dominate"

    # --- text (free text -> embedding) ---
    text: str                  # functional_text_plain: the card's rules text
    type_text: str             # human-readable type line, e.g. "Wizard Action - Attack"

    # --- misc / display ---
    played_horizontally: bool
    image_url: str | None      # first printing's image (for the Phase 6/7 UI)
    set_ids: list[str]         # sets this card has appeared in

    legality: Legality

    # ---- derived type facets -------------------------------------------------
    @cached_property
    def classes(self) -> list[str]:
        return [t for t in self.types if t in CLASSES]

    @cached_property
    def talents(self) -> list[str]:
        return [t for t in self.types if t in TALENTS]

    @cached_property
    def categories(self) -> list[str]:
        return [t for t in self.types if t in CARD_CATEGORIES]

    @cached_property
    def other_types(self) -> list[str]:
        """Recognized-as-nothing-above: subtypes (1H, Staff, Legs) and creature
        traits (Pirate, Dragon, Young) we haven't faceted yet. Kept for safety."""
        known = CLASSES | TALENTS | CARD_CATEGORIES | {"Generic"}
        return [t for t in self.types if t not in known]

    @property
    def is_generic(self) -> bool:
        return "Generic" in self.types

    @property
    def is_hero(self) -> bool:
        return "Hero" in self.types

    @property
    def is_equipment(self) -> bool:
        return "Equipment" in self.types

    @property
    def is_weapon(self) -> bool:
        return "Weapon" in self.types

    # ---- the text we will embed ---------------------------------------------
    @property
    def text_for_embedding(self) -> str:
        """Compose the document we embed for semantic search.

        Deliberately TEXT only — name, type line, keywords, traits, rules text —
        i.e. "what this card is and does". We intentionally EXCLUDE numeric stats
        (pitch/cost/power/...): embeddings reason poorly about exact numbers, and
        those belong to the structured filter instead. (The `search_document:`
        prefix nomic-embed-text requires is added later, in the embedding layer.)
        """
        parts = [self.name, self.type_text]
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords))
        if self.traits:
            parts.append("Traits: " + ", ".join(self.traits))
        if self.text:
            parts.append(expand_symbols(self.text))  # {h} -> life, {p} -> power, ...
        return "\n".join(parts)

    # ---- construction from a raw card.json record ---------------------------
    @classmethod
    def from_raw(cls, raw: dict) -> "Card":
        printings = raw.get("printings", [])
        return cls(
            unique_id=raw["unique_id"],
            name=raw["name"],
            color=(raw.get("color") or None),
            pitch=parse_stat(raw.get("pitch")),
            cost=parse_stat(raw.get("cost")),
            power=parse_stat(raw.get("power")),
            defense=parse_stat(raw.get("defense")),
            health=parse_stat(raw.get("health")),
            intelligence=parse_stat(raw.get("intelligence")),
            arcane=parse_stat(raw.get("arcane")),
            types=raw.get("types", []),
            traits=raw.get("traits", []),
            keywords=raw.get("card_keywords", []),
            text=raw.get("functional_text_plain", ""),
            type_text=raw.get("type_text", ""),
            played_horizontally=raw.get("played_horizontally", False),
            image_url=(printings[0].get("image_url") if printings else None) or None,
            set_ids=sorted({p["set_id"] for p in printings if p.get("set_id")}),
            legality=Legality(
                cc_legal=raw.get("cc_legal", False),
                cc_banned=raw.get("cc_banned", False),
                cc_suspended=raw.get("cc_suspended", False),
                cc_living_legend=raw.get("cc_living_legend", False),
                blitz_legal=raw.get("blitz_legal", False),
                blitz_banned=raw.get("blitz_banned", False),
                blitz_suspended=raw.get("blitz_suspended", False),
                blitz_living_legend=raw.get("blitz_living_legend", False),
                commoner_legal=raw.get("commoner_legal", False),
                commoner_banned=raw.get("commoner_banned", False),
                commoner_suspended=raw.get("commoner_suspended", False),
                ll_legal=raw.get("ll_legal", False),
                ll_banned=raw.get("ll_banned", False),
                ll_restricted=raw.get("ll_restricted", False),
                silver_age_legal=raw.get("silver_age_legal", False),
                silver_age_banned=raw.get("silver_age_banned", False),
                upf_banned=raw.get("upf_banned", False),
            ),
        )


def load_cards(path: Path = RAW_CARD_FILE) -> list[Card]:
    """Load and validate every card from card.json into typed Card objects."""
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Card.from_raw(r) for r in records]


if __name__ == "__main__":
    # Quick self-check: load everything and print a faceted view of one card.
    cards = load_cards()
    print(f"Loaded {len(cards):,} cards.\n")

    sample = next(c for c in cards if c.name == "Command and Conquer")
    print(f"Name:        {sample.name}")
    print(f"Color/Pitch: {sample.color} / {sample.pitch}")
    print(f"Stats:       cost={sample.cost} power={sample.power} defense={sample.defense}")
    print(f"Classes:     {sample.classes}")
    print(f"Talents:     {sample.talents}")
    print(f"Categories:  {sample.categories}")
    print(f"Other types: {sample.other_types}")
    print(f"Legality:    {sample.legality.by_format}")
    print(f"is_legal('Classic Constructed') -> {sample.legality.is_legal('Classic Constructed')}")
    print("\n--- text_for_embedding ---")
    print(sample.text_for_embedding)

    # Per-format legal counts across the whole set.
    print("\n--- legal cards per format ---")
    for fmt in ("cc", "blitz", "commoner", "ll", "silver_age", "upf"):
        n = sum(1 for c in cards if c.legality.is_legal(fmt))
        print(f"  {fmt:12} {n:,}")
