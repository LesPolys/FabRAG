"""Command-line front door — Phase 3.

Two ways in, sharing one set of structured filters:

  fabrag search "gain life when I attack" --class Wizard --legal cc -k 10
      -> raw retrieval: ranked cards and/or rules, with similarity scores.
         Fast, no LLM. Use it to see/trust what semantic search is finding.

  fabrag ask "what happens when an attack is defended?" --source rules
      -> full RAG: retrieve, then a grounded natural-language answer streamed
         live, followed by the cards/rules it was grounded in.

Both take --source cards|rules|all (default all): cards come from card.json,
rules from the chunked Comprehensive Rules + keyword glossary (rules.py).

This is deliberately thin — all the real work lives in retrieval.py / rag.py.
The CLI's only jobs are parsing flags into a CardFilter and printing nicely.
"""

from __future__ import annotations

import argparse
import sys

from . import rag
from .cards import Card
from .deck import find_hero, hero_pool_predicate
from .rag import answer_stream
from .retrieval import CardFilter, SearchResult


# --- filters: shared across both subcommands ---------------------------------
def _add_filter_args(p: argparse.ArgumentParser) -> None:
    """Attach the structured-filter flags (the metadata half of hybrid search)."""
    p.add_argument("query", help="natural-language query")
    p.add_argument("-k", "--top-k", type=int, default=8, dest="k",
                   help="number of results to retrieve (default: 8)")
    p.add_argument("--source", choices=("cards", "rules", "all"), default="all",
                   help="which corpora to search (default: all)")
    p.add_argument("--color", help="pitch color: Red / Yellow / Blue")
    p.add_argument("--pitch", type=int, help="exact pitch value")
    p.add_argument("--cost", type=int, help="exact resource cost")
    p.add_argument("--class", action="append", dest="classes", metavar="CLASS",
                   help="class, e.g. Wizard (repeatable, any-of)")
    p.add_argument("--talent", action="append", dest="talents", metavar="TALENT",
                   help="talent, e.g. Ice (repeatable, any-of)")
    p.add_argument("--category", action="append", dest="categories", metavar="CAT",
                   help="card category, e.g. Attack (repeatable, any-of)")
    p.add_argument("--keyword", action="append", dest="keywords", metavar="KW",
                   help="keyword, e.g. 'Go Again' (repeatable, any-of)")
    p.add_argument("--legal", dest="legal_in", metavar="FORMAT",
                   help="restrict to a legal format: cc / blitz / commoner / ll / silver_age / upf")
    p.add_argument("--hero", metavar="NAME",
                   help="restrict to a hero's legal deck pool (class/talent/legality); "
                        "uses --legal's format, else cc")


def _filter_from_args(args: argparse.Namespace) -> CardFilter:
    return CardFilter(
        color=args.color,
        pitch=args.pitch,
        cost=args.cost,
        classes=tuple(args.classes or ()),
        talents=tuple(args.talents or ()),
        categories=tuple(args.categories or ()),
        keywords=tuple(args.keywords or ()),
        legal_in=args.legal_in,
    )


# --- pretty-printing ---------------------------------------------------------
def _stat_line(card) -> str:
    bits = []
    if card.color:
        bits.append(card.color)
    if card.pitch is not None:
        bits.append(f"pitch {card.pitch}")
    if card.cost is not None:
        bits.append(f"cost {card.cost}")
    if card.power is not None:
        bits.append(f"power {card.power}")
    if card.defense is not None:
        bits.append(f"def {card.defense}")
    return "  (" + ", ".join(bits) + ")" if bits else ""


def _print_results(results: list[SearchResult]) -> None:
    for i, r in enumerate(results, start=1):
        d = r.doc
        if isinstance(d, Card):
            print(f"{i:>2}. [{r.score:.3f}] {d.name} — {d.type_text}{_stat_line(d)}")
        else:  # RuleChunk — cite it, locate it, and link it for verification
            where = f" — {d.section}" if d.kind == "rule" else ""
            url = f"  <{d.source_url}>" if d.source_url else ""
            print(f"{i:>2}. [{r.score:.3f}] {d.citation}{where}{url}")


# --- hero pool ---------------------------------------------------------------
def _resolve_hero_predicate(args: argparse.Namespace):
    """If --hero was given, return (predicate, banner); else (None, None).

    Resolves the hero from the shared corpus and builds a pool predicate for the
    chosen format. Raises ValueError (caught by the caller) on a bad hero name.
    """
    if not args.hero:
        return None, None
    # The shared corpus may mix Cards and RuleChunks; heroes live in the cards.
    cards = [d for d in rag.get_retriever().docs if isinstance(d, Card)]
    hero = find_hero(args.hero, cards)
    fmt = args.legal_in or "cc"
    talents = ", ".join(hero.talents) or "none"
    banner = (f"Hero: {hero.name} — {'/'.join(hero.classes) or 'classless'} "
              f"(talents: {talents}); pool format: {fmt}\n")
    return hero_pool_predicate(hero, fmt), banner


# --- subcommands -------------------------------------------------------------
_PITCH_LABEL = {1: "Pitch 1 (red)", 2: "Pitch 2 (yellow)", 3: "Pitch 3 (blue)"}


def _cmd_build(args: argparse.Namespace) -> int:
    """fabrag build: hero + strategy -> a guaranteed-legal registration + rationale."""
    from collections import Counter

    from .build import build_deck, explain_deck_stream
    from .formats import get_format

    try:
        result = build_deck(args.hero, args.strategy, args.format)
    except ValueError as e:  # bad hero name / ambiguous match / wrong-age hero
        print(e, file=sys.stderr)
        return 1

    pool = result.pool
    rules = get_format(pool.format)
    total = len(pool.all_cards)
    cap = f"/{rules.pool_max}" if rules.pool_max else ""
    print(f"Hero:     {pool.hero.name} ({pool.hero.type_text})")
    print(f"Format:   {rules.label}   |   strategy: {args.strategy}")
    print(f"Sourcing: {', '.join(result.queries)}")
    print(f"Rounds:   {result.rounds}"
          + (f"   |   finisher: {'; '.join(result.finisher_notes)}" if result.finisher_notes else ""))
    print(f"Legal:    {result.validation.ok}   |   pool: {total}{cap} registered cards")

    # 1. Inventory — the equipped weapons & equipment, by slot.
    if pool.inventory:
        print(f"\nInventory ({len(pool.inventory)}):")
        for c in sorted(pool.inventory, key=lambda c: (c.equipment_slot or "Weapon", c.name)):
            slot = c.equipment_slot or "Weapon"
            print(f"  [{slot}] {c.name} — {c.type_text}")

    # 2. Starting deck — grouped by pitch.
    groups: dict[str, list[tuple[int, str]]] = {}
    counts = Counter((c.name, c.pitch) for c in pool.deck)
    by_key = {(c.name, c.pitch): c for c in pool.deck}
    for (name, pitch), n in sorted(counts.items()):
        c = by_key[(name, pitch)]
        groups.setdefault(_PITCH_LABEL.get(pitch, "Other"), []).append((n, f"{name} — {c.type_text}"))
    print(f"\nStarting deck ({len(pool.deck)} cards):")
    for label in ["Pitch 1 (red)", "Pitch 2 (yellow)", "Pitch 3 (blue)", "Other"]:
        if label in groups:
            print(f"  {label}:")
            for n, line in groups[label]:
                print(f"    {n}x {line}")

    # 3. Sideboard — the LLM's matchup suggestions, with rationale.
    if result.sideboard_notes:
        print(f"\nSideboard ({len(pool.sideboard)} suggested):")
        for note in result.sideboard_notes:
            print(f"  {note}")

    # 4. Stats — the deck's pitch / cost / type shape.
    from .stats import deck_stats, render_stats
    print("\n" + render_stats(deck_stats(pool)))

    for w in result.warnings:
        print(f"\n  note: {w}")

    if not args.no_explain:
        print("\nGame plan:")
        for piece in explain_deck_stream(result, args.strategy):
            print(piece, end="", flush=True)
        print()
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    """fabrag chat: build a deck, then talk about it — grounded in the hero's
    legal pool + the rules corpus, streamed turn by turn."""
    from .build import build_deck
    from .chat import deck_chat_stream, pool_to_text

    try:
        result = build_deck(args.hero, args.strategy, args.format)
    except ValueError as e:  # bad hero name / wrong-age hero
        print(e, file=sys.stderr)
        return 1

    pool = result.pool
    print(f"Built {pool.hero.name} ({pool.format}) — {len(pool.deck)} cards, "
          f"{len(pool.inventory)} equipped, {len(pool.sideboard)} sideboard.")
    print("Ask about the deck; blank line or Ctrl-D to quit.")
    deck_text = pool_to_text(pool)
    history: list[dict] = []
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            break
        _results, stream = deck_chat_stream(pool.hero, pool.format, deck_text, history, question)
        answer = ""
        for piece in stream:
            print(piece, end="", flush=True)
            answer += piece
        print()
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """fabrag serve: the web UI — a thin FastAPI skin over the same pipeline."""
    import uvicorn

    print(f"FabRAG web UI -> http://{args.host}:{args.port}")
    uvicorn.run("fabrag.web.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run the gold-set retrieval eval (and optionally the grounding eval).

    Retrieval is cheap (one query embedding per case); grounding is slow (one
    or two chat-model calls per question), so it's opt-in via --grounding N.
    """
    from .evaluation import evaluate, load_gold_set

    cases = load_gold_set()
    print(evaluate(cases, k=args.k, source=args.eval_source, mode=args.mode).render())

    if args.grounding:
        from .grounding import grounding_eval, render_grounding

        questions = [c.query for c in cases][: args.grounding]
        print()
        print(render_grounding(
            grounding_eval(questions, k=args.k, use_judge=not args.no_judge)
        ))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    filters = _filter_from_args(args)
    try:
        predicate, banner = _resolve_hero_predicate(args)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if banner:
        print(banner)
    results = rag.retrieve(
        args.query, k=args.k, source=args.source, filters=filters, predicate=predicate
    )
    if not results:
        print("Nothing matched your query and filters.")
        return 0
    _print_results(results)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    filters = _filter_from_args(args)
    try:
        predicate, banner = _resolve_hero_predicate(args)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if banner:
        print(banner)
    results, stream = answer_stream(
        args.query, k=args.k, source=args.source, filters=filters, predicate=predicate
    )
    for piece in stream:        # tokens arrive live
        print(piece, end="", flush=True)
    print()
    if results:
        print("\nGrounded in:")
        _print_results(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    # FAB card text (and our em-dash separator) is non-ASCII; the default
    # Windows console is cp1252 and would mangle it. Force UTF-8 output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="fabrag",
        description="Local RAG deck-building assistant for Flesh and Blood.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="ranked card retrieval (no LLM)")
    _add_filter_args(p_search)
    p_search.set_defaults(func=_cmd_search)

    p_ask = sub.add_parser("ask", help="ask a question, get a grounded answer")
    _add_filter_args(p_ask)
    p_ask.set_defaults(func=_cmd_ask)

    p_build = sub.add_parser("build", help="generate a legal deck for a hero + strategy")
    p_build.add_argument("--hero", required=True, metavar="NAME",
                         help="hero to build for (partial names ok if unambiguous)")
    p_build.add_argument("--strategy", default="a balanced, efficient deck",
                         help="deck strategy/archetype in plain words")
    p_build.add_argument("--format", default="cc", dest="format",
                         help="format: cc / blitz / commoner (default: cc)")
    p_build.add_argument("--no-explain", action="store_true",
                         help="skip the LLM game-plan explanation")
    p_build.set_defaults(func=_cmd_build)

    p_chat = sub.add_parser("chat", help="build a deck, then chat about it (grounded)")
    p_chat.add_argument("--hero", required=True, metavar="NAME",
                        help="hero to build for (partial names ok if unambiguous)")
    p_chat.add_argument("--strategy", default="a balanced, efficient deck",
                        help="deck strategy/archetype in plain words")
    p_chat.add_argument("--format", default="cc", dest="format",
                        help="format: cc / blitz / commoner (default: cc)")
    p_chat.set_defaults(func=_cmd_chat)

    p_serve = sub.add_parser("serve", help="start the web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_eval = sub.add_parser("eval", help="run the gold-set evaluation suite")
    p_eval.add_argument("-k", "--top-k", type=int, default=8, dest="k",
                        help="retrieval depth to evaluate at (default: 8)")
    p_eval.add_argument("--source", choices=("all", "per-kind", "cards", "rules"),
                        default="per-kind", dest="eval_source",
                        help="'all' = production pipeline; 'per-kind' = each case "
                             "vs its own corpus, the clean ranking lens (default)")
    p_eval.add_argument("--mode", choices=("hybrid", "dense", "lexical"),
                        default="hybrid", help="ranking mode under test (default: hybrid)")
    p_eval.add_argument("--grounding", type=int, metavar="N", default=0,
                        help="also grade N generated answers for grounding (slow)")
    p_eval.add_argument("--no-judge", action="store_true",
                        help="grounding: skip the LLM judge, citation audit only")
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
