"""Quota-split sweep — Phase 8: how much of a mixed top-k should be rules?

rag.retrieve reserves a fraction of the budget for rules when source="all"
(RULES_SHARE, shipped as a 0.3 guess). This sweeps that fraction over the
FULL gold set on the production pipeline — every case retrieved with
source="all", exactly what a `fabrag ask` user gets. The tension being
measured: card cases lose recall as their quota shrinks; rule cases lose it
the other way. Overall recall@k arbitrates, with the per-kind columns
showing who pays for each setting.

Run:  uv run python scripts/sweep_quota.py    (fast: index is cached)
"""

from __future__ import annotations

from fabrag.evaluation import EvalReport, load_gold_set, score_case
from fabrag.rag import retrieve

SHARES = [0.15, 0.25, 0.30, 0.40, 0.50]
K = 8


def main() -> None:
    cases = load_gold_set()
    print(f"{len(cases)} cases, k={K}, source=all (production pipeline)\n")
    print(f"{'share':>6} {'k_rules':>8} {'all r@8':>8} {'cards r@8':>10} {'rules r@8':>10}")

    for share in SHARES:
        results = [
            score_case(c, retrieve(c.query, k=K, source="all", rules_share=share))
            for c in cases
        ]
        report = EvalReport(results, k=K, source=f"all/{share}")
        k_rules = max(2, round(K * share))
        s_all = report.summary()
        s_c = report.summary("card")
        s_r = report.summary("rule")
        print(
            f"{share:>6.2f} {k_rules:>8} {s_all[f'recall@{K}']:>8.3f} "
            f"{s_c[f'recall@{K}']:>10.3f} {s_r[f'recall@{K}']:>10.3f}"
        )


if __name__ == "__main__":
    main()
