"""Embedding helpers — Phase 2, the 'E' that makes retrieval possible.

An embedding maps text to a vector (768 numbers for nomic-embed-text) such that
texts with similar meaning land close together. We run the model locally via
Ollama.

Two things this module gets right, both learned the hard way in Phase 0:
  1. nomic-embed-text REQUIRES task prefixes: 'search_document: ' for the things
     we store, 'search_query: ' for the question. Skipping them collapses
     similarity into a mushy band. We bake them in so callers can't forget.
  2. We L2-normalize every vector to unit length. Then cosine similarity ==
     dot product, so scoring a query against the whole corpus is one matrix
     multiply (see retrieval.py).
"""

from __future__ import annotations

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768  # nomic-embed-text output dimensionality

# Prefixes the model was trained with (see memory: nomic-embed-prefixes).
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product == cosine similarity.

    Guards against divide-by-zero on the (theoretical) all-zero vector.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _embed(texts: list[str], prefix: str, batch_size: int = 128) -> np.ndarray:
    """Embed a list of texts (with the given task prefix) into a normalized matrix.

    Returns an array of shape (len(texts), EMBED_DIM), float32, unit-normalized.
    Batches the Ollama calls so we don't send one giant request.
    """
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start : start + batch_size]]
        resp = ollama.embed(model=EMBED_MODEL, input=batch)
        out.append(np.asarray(resp.embeddings, dtype=np.float32))
    matrix = np.vstack(out) if out else np.zeros((0, EMBED_DIM), dtype=np.float32)
    return _normalize(matrix)


def embed_documents(texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Embed corpus documents (the cards we store). Shape: (n, EMBED_DIM)."""
    return _embed(texts, _DOC_PREFIX, batch_size)


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query. Shape: (EMBED_DIM,)."""
    return _embed([text], _QUERY_PREFIX)[0]
