"""Reranker probe — Phase 8 stretch: can an LLM judge rescue the hard cases?

The gold set's known-hard card cases need CO-OCCURRENCE STRUCTURE: "destroy
an equipment WHEN an attack HITS" — every word common, only the relationship
rare. BM25 is bag-of-words (can't see relationships); the embedding measurably
doesn't either. An LLM reading the actual card text CAN. The standard
production answer is a cross-encoder reranker; before building a stage, this
probe asks whether even a cheap local-LLM judge moves the needle:

    top-30 hybrid results -> one qwen call: "which of these actually match?"
    -> relevant cards float to the front -> where do the gold answers land?

This is a PROBE (2 queries, 1 LLM call each), not an eval — promising numbers
here justify building + properly evaluating a reranker stage later.

Run:  uv run python scripts/probe_reranker.py
"""

from __future__ import annotations

import json

import ollama

from fabrag.cards import expand_symbols
from fabrag.evaluation import load_gold_set, matches
from fabrag.generation import CHAT_MODEL, NUM_CTX
from fabrag.rag import retrieve

HARD_CASE_IDS = ["card-destroy-equipment-on-hit", "card-freeze-arsenal"]
DEPTH = 30  # rerank window: deep enough to contain the gold answers

_PROMPT = """\
You judge search results for a Flesh and Blood card search engine. Given a
QUERY and numbered CARDS, list the numbers of every card that actually
satisfies the query — read each card's text carefully and apply ALL the
query's conditions, not just keyword overlap. Respond with JSON only:
{"relevant": [3, 17, ...]}\
"""


def rerank(query: str, results) -> list:
    sheet = "\n".join(
        f"{i}. {r.doc.name}: {expand_symbols(r.doc.text)[:220]}"
        for i, r in enumerate(results, start=1)
    )
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": f"QUERY: {query}\n\nCARDS:\n{sheet}"},
        ],
        options={"temperature": 0.0, "num_ctx": NUM_CTX},
        format="json",
    )
    try:
        picked = {int(n) for n in json.loads(resp.message.content).get("relevant", [])}
    except (json.JSONDecodeError, TypeError, ValueError):
        picked = set()
    # Stable partition: judged-relevant first (original order), rest after.
    yes = [r for i, r in enumerate(results, start=1) if i in picked]
    no = [r for i, r in enumerate(results, start=1) if i not in picked]
    return yes + no


def gold_ranks(case, results) -> dict[str, int | None]:
    return {
        e: next((i for i, r in enumerate(results, start=1) if matches(e, r)), None)
        for e in case.expected
    }


def main() -> None:
    cases = {c.id: c for c in load_gold_set()}
    for cid in HARD_CASE_IDS:
        case = cases[cid]
        before = retrieve(case.query, k=DEPTH, source="cards")
        after = rerank(case.query, before)
        print(f"{cid}: {case.query!r}")
        for e, b in gold_ranks(case, before).items():
            a = gold_ranks(case, after)[e]
            fmt = lambda r: f"#{r}" if r else f">{DEPTH}"
            print(f"  {e:22} hybrid {fmt(b):>5}  ->  reranked {fmt(a):>5}")
        print()


if __name__ == "__main__":
    main()
