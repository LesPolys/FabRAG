"""Chunk-size sweep — Phase 8: is 1500/3000 chars actually the right knob?

rules.py shipped with target=1500 / max=3000 as an informed guess. This sweep
re-chunks the CR at several sizes, embeds each variant fresh (NO disk cache —
we don't want six stale .npz files), and scores the RULE cases of the gold
set per-kind. Smaller chunks = sharper topical focus but less context per
hit; bigger = more context but blurrier vectors. The gold set decides.

Run:  uv run python scripts/sweep_chunks.py        (~2-3 min: re-embeds per config)
"""

from __future__ import annotations

from fabrag.evaluation import EvalReport, load_gold_set, score_case
from fabrag.retrieval import Retriever, card_retriever, scope
from fabrag.rules import load_keyword_chunks, load_rule_chunks

# (target_chars, max_chars) — max = 2x target, matching the shipped ratio.
CONFIGS = [(600, 1200), (1000, 2000), (1500, 3000), (2200, 4400), (3000, 6000)]
K = 8


def main() -> None:
    rule_cases = [c for c in load_gold_set() if c.kind == "rule"]
    keywords = load_keyword_chunks()  # not size-dependent; reused across configs
    cards = card_retriever()          # constant across configs; cached on disk

    # MEASURE THE PRODUCTION PATH. A first version of this sweep built a
    # rules-only retriever — but production searches the MERGED corpus, where
    # BM25's idf and length statistics include the cards. The rules-only
    # numbers said 2200/4400 beat 1500/3000; the merged numbers disagreed.
    # Evaluating a flattering simplification is how that sweep lied.
    print(f"{len(rule_cases)} rule cases, k={K}, merged corpus + scope('rules')\n")
    print(f"{'target/max':>12} {'chunks':>7} {'recall@8':>9} {'mrr':>7} {'ndcg@8':>8}")

    for target, mx in CONFIGS:
        chunks = load_rule_chunks(target_chars=target, max_chars=mx) + keywords
        retriever = Retriever.merge(cards, Retriever.build(chunks))
        predicate = scope("rules")
        results = [
            score_case(c, retriever.search(c.query, k=K, predicate=predicate))
            for c in rule_cases
        ]
        report = EvalReport(results, k=K, source=f"rules/{target}")
        s = report.summary()
        print(
            f"{target:>6}/{mx:<5} {len(chunks):>7} "
            f"{s[f'recall@{K}']:>9.3f} {s['mrr']:>7.3f} {s[f'ndcg@{K}']:>8.3f}"
        )


if __name__ == "__main__":
    main()
