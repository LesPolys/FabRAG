"""Command-line front door — Phase 3.

Two ways in, sharing one set of structured filters:

  fabrag search "gain life when I attack" --class Wizard --legal cc -k 10
      -> raw retrieval: the ranked cards, with similarity scores. Fast, no LLM.
         Use it to see/trust what semantic search is actually finding.

  fabrag ask "which cards deal arcane damage to the hero?" --color Blue
      -> full RAG: retrieve, then a grounded natural-language answer streamed
         live, followed by the cards it was grounded in.

This is deliberately thin — all the real work lives in retrieval.py / rag.py.
The CLI's only jobs are parsing flags into a CardFilter and printing nicely.
"""

from __future__ import annotations

import argparse
import sys

from .rag import answer_stream
from .retrieval import CardFilter, Retriever, SearchResult


# --- filters: shared across both subcommands ---------------------------------
def _add_filter_args(p: argparse.ArgumentParser) -> None:
    """Attach the structured-filter flags (the metadata half of hybrid search)."""
    p.add_argument("query", help="natural-language query")
    p.add_argument("-k", "--top-k", type=int, default=8, dest="k",
                   help="number of cards to retrieve (default: 8)")
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
        print(f"{i:>2}. [{r.score:.3f}] {r.card.name} — {r.card.type_text}"
              f"{_stat_line(r.card)}")


# --- subcommands -------------------------------------------------------------
def _cmd_search(args: argparse.Namespace) -> int:
    filters = _filter_from_args(args)
    retriever = Retriever.load_or_build()
    results = retriever.search(args.query, k=args.k, filters=filters)
    if not results:
        print("No cards matched your query and filters.")
        return 0
    _print_results(results)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    filters = _filter_from_args(args)
    results, stream = answer_stream(args.query, k=args.k, filters=filters)
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
