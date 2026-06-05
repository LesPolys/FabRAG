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

import os

import numpy as np
import ollama

# Each embedding model is trained with its OWN task-prefix convention — using
# the wrong one (or none) quietly degrades retrieval, which is exactly the
# kind of failure only the eval harness would catch. The registry keeps
# (dimensionality, doc prefix, query prefix) next to the model name so a swap
# can't mix conventions.
_MODELS: dict[str, tuple[int, str, str]] = {
    # nomic: symmetric task prefixes on both sides.
    "nomic-embed-text": (768, "search_document: ", "search_query: "),
    # mxbai: instruction on the QUERY side only; documents are embedded bare.
    "mxbai-embed-large": (
        1024,
        "",
        "Represent this sentence for searching relevant passages: ",
    ),
}

# An env var rather than a function parameter: the model choice must be ONE
# global fact — retriever indexes, fingerprints, and query embedding all have
# to agree, and threading a parameter through every layer invites a mismatch.
#   FABRAG_EMBED_MODEL=nomic-embed-text uv run python -m fabrag.evaluation
# Default is mxbai-embed-large: on the gold set it beats nomic-embed-text on
# every metric (cards recall@8 0.292 vs 0.042 dense; 0.625 vs 0.458 hybrid).
EMBED_MODEL = os.environ.get("FABRAG_EMBED_MODEL", "mxbai-embed-large")
if EMBED_MODEL not in _MODELS:
    raise ValueError(f"Unknown embed model {EMBED_MODEL!r}; known: {sorted(_MODELS)}")
EMBED_DIM, _DOC_PREFIX, _QUERY_PREFIX = _MODELS[EMBED_MODEL]


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
