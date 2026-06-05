"""Lexical scoring — BM25, hand-rolled. Phase 5's eval-driven fix.

The eval harness measured what demos hid: dense embeddings alone collapse on
compositional card queries (cards recall@8 = 0.04 at baseline). "destroy an
equipment when an attack hits" is three precise constraints; a 768-dim vector
smears them into one fuzzy direction, and "hub" cards that are vaguely close
to everything outrank the card that literally says those words. Embeddings
know what text MEANS; they're bad at insisting on what it SAYS.

Lexical search is the complement: score documents by the exact terms they
share with the query. The classic scorer is BM25 — tf-idf, refined:

    score(q, d) = Σ_{t in q}  idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·norm(d))

  - idf(t): rare terms count more. "arsenal" appears in ~50 docs and is worth
    a lot; "card" appears everywhere and is worth nearly nothing. (This is
    also why we don't bother stripping stopwords — idf zeroes them for free.)
  - tf saturation (k1): the 2nd occurrence of "arsenal" in a doc is worth less
    than the 1st, the 10th nearly nothing — k1 controls how fast the curve
    flattens. Plain tf-idf rewards keyword-stuffing linearly; BM25 doesn't.
  - length normalization (b): norm(d) = 1 - b + b·(len(d)/avg_len). Long docs
    match more terms by accident, so their tf is discounted; b dials that
    correction from 0 (off) to 1 (full).

k1=1.5, b=0.75 are the canonical defaults; they go unquestioned here until
the gold set says otherwise.

We index the SAME text the dense side embeds (text_for_embedding), so the two
scorers are different lenses on one corpus, and fusing their rankings
(retrieval.py) is apples-to-apples.
"""

from __future__ import annotations

import re
from collections import Counter
from math import log

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs. Deliberately dumb — no stemming, no
    stopwords. BM25's idf already mutes common words, and FAB vocabulary is
    mostly exact terms of art ("arsenal", "dominate") where stemming would
    only blur. Complexity here must be paid for by the eval."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """An inverted index with BM25 scoring over a fixed corpus.

    Build is cheap (~5k docs in well under a second), so unlike the embedding
    matrix it isn't persisted — we rebuild from the docs at load time.
    """

    def __init__(self, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(texts)

        doc_tokens = [tokenize(t) for t in texts]
        self.doc_len = np.array([len(toks) for toks in doc_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0

        # Postings: term -> (doc row indices, term frequencies in those docs).
        # Stored as arrays so scoring a term is one vectorized accumulate.
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        per_term: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(doc_tokens):
            for term, tf in Counter(toks).items():
                per_term.setdefault(term, []).append((i, tf))
        for term, pairs in per_term.items():
            rows = np.array([p[0] for p in pairs], dtype=np.intp)
            tfs = np.array([p[1] for p in pairs], dtype=np.float32)
            self.postings[term] = (rows, tfs)

        # Robust ("Lucene-style") idf: ln(1 + (N - df + 0.5)/(df + 0.5)).
        # Always positive, unlike the textbook form which can go negative for
        # terms in more than half the corpus.
        self.idf = {
            term: log(1.0 + (self.n_docs - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, (rows, _) in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document against `query`. Shape: (n_docs,).

        Zero for documents sharing no terms with the query — meaningfully
        zero, unlike cosine similarity where ~0.5 can still mean "unrelated".
        """
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return out
        norm = 1.0 - self.b + self.b * (self.doc_len / self.avg_len)
        for term in tokenize(query):
            posting = self.postings.get(term)
            if posting is None:
                continue  # term absent from the corpus contributes nothing
            rows, tf = posting
            out[rows] += self.idf[term] * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm[rows])
        return out
