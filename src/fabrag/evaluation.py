"""Evaluation harness — Phase 5, measuring retrieval before trusting it.

Up to now "does retrieval work?" has been vibes: run a query, eyeball the top
results. That stops scaling the moment we want to TUNE anything — chunk size,
the card/rule quota split, k, embedding prefixes — because a change that helps
one query can silently hurt ten others. The fix is the oldest idea in IR:

  1. A GOLD SET of hand-authored cases: a query plus the documents that
     truly answer it (ground truth established by exact text search and CR
     lookup — never by semantic search, which would bias the gold set toward
     whatever the current retriever already finds).
  2. RANK METRICS over the gold set:
       - recall@k : of the expected documents, what fraction shows up in the
                    top k? (Did we find it at all?)
       - MRR      : 1/rank of the first relevant hit, averaged. (How high?)
       - nDCG@k   : like recall but rank-discounted — a hit at rank 1 is worth
                    more than a hit at rank 8, on a 1/log2 curve. (How well
                    ordered?)

Every metric is computed against the SAME pipeline users hit (rag.retrieve,
quotas included) — we evaluate the system, not a flattering simplification.

Robustness rule: gold cases never reference chunk_ids. Chunk boundaries move
whenever we tune chunking (that's the point of tuning), so expectations bind
to stable identities instead — card NAMES, and CR rule NUMBERS matched against
each chunk's rule_ids. A re-chunk changes which chunk carries rule 7.3.2, but
"7.3.2 answers this question" stays true.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import log2
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel

from .cards import Card
from .retrieval import Mode, Retriever, SearchResult, Source
from .rules import RuleChunk

# Hand-authored, version-controlled (unlike data/raw — this is NOT regenerable).
GOLD_SET_PATH = Path(__file__).resolve().parents[2] / "data" / "gold" / "gold_set.json"


# =============================================================================
# Gold cases
# =============================================================================
class GoldCase(BaseModel):
    """One evaluation case: a query and the documents that truly answer it.

    `expected` entries by kind:
      kind="card":  exact card names, e.g. "Sink Below" (matches every pitch
                    version of that name — they share it).
      kind="rule":  stable rule references —
                    "cr7.3.2"       a CR rule/subrule number (matches any chunk
                                    whose rule_ids contain it)
                    "cr7.3.*"       a section prefix (any rule under 7.3)
                    "kw:Dominate"   a keyword.json entry by name
                    "gloss:Dominate" a CR glossary term by name

    `match` decides what multiple entries mean:
      "all" (default): every entry is relevant — recall measures the fraction
                       retrieved ("which cards destroy equipment on hit?" has
                       three true answers; finding one of three is 1/3).
      "any":           the entries are alternatives — finding any one fully
                       answers the question ("what does dominate do?" is
                       answered by the keyword entry OR the glossary entry;
                       missing one of them is not a miss).
    """

    id: str
    query: str
    kind: Literal["card", "rule"]
    expected: list[str]
    match: Literal["all", "any"] = "all"
    notes: str = ""


def load_gold_set(path: Path = GOLD_SET_PATH) -> list[GoldCase]:
    records = json.loads(path.read_text(encoding="utf-8"))
    cases = [GoldCase(**r) for r in records]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate gold case ids: {dupes}")
    return cases


# =============================================================================
# Matching: does a retrieved document satisfy an expected entry?
# =============================================================================
def matches(expected: str, result: SearchResult) -> bool:
    doc = result.doc
    if expected.startswith("kw:"):
        return (
            isinstance(doc, RuleChunk)
            and doc.kind == "keyword"
            and (doc.title or "").lower() == expected[3:].lower()
        )
    if expected.startswith("gloss:"):
        return (
            isinstance(doc, RuleChunk)
            and doc.kind == "glossary"
            and (doc.title or "").lower() == expected[6:].lower()
        )
    if expected.startswith("cr"):
        if not isinstance(doc, RuleChunk):
            return False
        if expected.endswith(".*"):  # section prefix: cr7.3.* hits cr7.3.2a etc.
            prefix = expected[:-1]   # keep the trailing dot
            return any(rid.startswith(prefix) for rid in doc.rule_ids)
        return expected in doc.rule_ids
    # Otherwise: a card name.
    return isinstance(doc, Card) and doc.name.lower() == expected.lower()


# =============================================================================
# Per-case scoring
# =============================================================================
@dataclass(frozen=True)
class CaseResult:
    case: GoldCase
    ranks: dict[str, int | None]  # expected entry -> best (1-based) rank, or None

    @property
    def first_rank(self) -> int | None:
        found = [r for r in self.ranks.values() if r is not None]
        return min(found) if found else None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_rank if self.first_rank else 0.0

    def recall_at(self, k: int) -> float:
        if self.case.match == "any":  # alternatives: any hit fully answers
            return 1.0 if self.first_rank is not None and self.first_rank <= k else 0.0
        hits = sum(1 for r in self.ranks.values() if r is not None and r <= k)
        return hits / len(self.ranks)

    def ndcg_at(self, k: int) -> float:
        """Binary-relevance nDCG: each expected entry credits at most one rank
        (its best), so three pitch-versions of one card can't triple-count.
        DCG sums 1/log2(rank+1) over credited ranks; IDCG is the same sum if
        every entry had landed in the top ranks — i.e. the perfect ordering.
        For match="any" only the best alternative counts and the ideal is a
        single rank-1 hit."""
        if self.case.match == "any":
            r = self.first_rank
            return 1.0 / log2(r + 1) if r is not None and r <= k else 0.0
        credited = sorted(r for r in self.ranks.values() if r is not None and r <= k)
        dcg = sum(1.0 / log2(r + 1) for r in credited)
        ideal_n = min(len(self.ranks), k)
        idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_n + 1))
        return dcg / idcg if idcg else 0.0


def score_case(case: GoldCase, results: list[SearchResult]) -> CaseResult:
    ranks: dict[str, int | None] = {}
    for expected in case.expected:
        ranks[expected] = next(
            (i for i, r in enumerate(results, start=1) if matches(expected, r)),
            None,
        )
    return CaseResult(case, ranks)


# =============================================================================
# Suite runner + report
# =============================================================================
@dataclass(frozen=True)
class EvalReport:
    results: list[CaseResult]
    k: int
    source: str

    def _subset(self, kind: str | None) -> list[CaseResult]:
        return [r for r in self.results if kind is None or r.case.kind == kind]

    def summary(self, kind: str | None = None) -> dict[str, float]:
        rs = self._subset(kind)
        if not rs:
            return {}
        return {
            "cases": len(rs),
            f"recall@{self.k}": mean(r.recall_at(self.k) for r in rs),
            "mrr": mean(r.reciprocal_rank for r in rs),
            f"ndcg@{self.k}": mean(r.ndcg_at(self.k) for r in rs),
        }

    def render(self) -> str:
        """Human-readable report: aggregate table, then every miss spelled out
        (a metric tells you THAT something regressed; the misses tell you WHAT)."""
        lines = [f"Retrieval eval — k={self.k}, source={self.source}, "
                 f"{len(self.results)} cases", ""]
        header = f"{'subset':10} {'cases':>5} {'recall@' + str(self.k):>10} {'mrr':>7} {'ndcg@' + str(self.k):>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for label, kind in (("all", None), ("cards", "card"), ("rules", "rule")):
            s = self.summary(kind)
            if s:
                lines.append(
                    f"{label:10} {s['cases']:>5} {s[f'recall@{self.k}']:>10.3f} "
                    f"{s['mrr']:>7.3f} {s[f'ndcg@{self.k}']:>8.3f}"
                )

        misses = [
            (r, e) for r in self.results
            for e, rank in r.ranks.items() if rank is None or rank > self.k
        ]
        if misses:
            lines.append("")
            lines.append(f"misses (not in top {self.k}):")
            for r, e in misses:
                rank = r.ranks[e]
                where = f"rank {rank}" if rank else "not retrieved"
                lines.append(f"  {r.case.id:28} {e:24} {where}  — {r.case.query!r}")
        return "\n".join(lines)


def evaluate(
    cases: list[GoldCase],
    *,
    k: int = 8,
    source: Source | Literal["per-kind"] = "all",
    mode: Mode = "hybrid",
    retriever: Retriever | None = None,
) -> EvalReport:
    """Run every case through the production retrieval path and score it.

    Two ways to read the system, and they answer different questions:
      source="all":      the production pipeline, quotas included. recall@k is
                         the headline ("did the LLM's context contain the
                         answer?"), but MRR/nDCG are polluted by an artifact —
                         retrieve() orders cards-then-rules, so a rule case's
                         best possible rank is wherever the rule quota starts.
      source="per-kind": each case searches only its own corpus (card cases vs
                         cards, rule cases vs rules). Ranks are now pure
                         within-source ranking quality — the right lens for
                         MRR/nDCG and for tuning chunking or embeddings.

    Imported lazily from rag to avoid a circular import (rag imports nothing
    from here; evaluation sits above the pipeline it measures).
    """
    from .rag import retrieve

    def src(case: GoldCase) -> Source:
        if source == "per-kind":
            return "cards" if case.kind == "card" else "rules"
        return source

    results = [
        score_case(
            case,
            retrieve(case.query, k=k, source=src(case), mode=mode, retriever=retriever),
        )
        for case in cases
    ]
    return EvalReport(results, k=k, source=f"{source}/{mode}")


if __name__ == "__main__":
    # The A/B that justifies (or kills) hybrid retrieval: same gold set, same
    # k, per-kind isolation — only the ranking mode varies.
    cases = load_gold_set()
    for mode in ("dense", "lexical", "hybrid"):
        print(evaluate(cases, source="per-kind", mode=mode).render())
        print()
