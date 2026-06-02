# FabRAG — a local AI deck builder for Flesh and Blood

FabRAG is a fully-local Retrieval-Augmented Generation (RAG) system that answers
questions about, and helps build decks for, the *Flesh and Blood* trading card game.
It runs entirely on your machine — no cloud APIs, no data leaving your computer.

It is also a learning project: each phase is built to teach a specific RAG/AI
fundamental, with the *why* behind every architecture decision documented as we go.

## Why this is more than "chat with your docs"

A deck builder can't rely on semantic search alone. "Show me legal blue Guardian
defense reactions" is a structured filter, not a similarity query. FabRAG therefore
combines three layers:

```
3. REASONING / GENERATION   local LLM explains synergies, assembles a legal deck
2. RETRIEVAL                hybrid: structured metadata filter  +  semantic vector search
1. DATA                     normalized, validated card objects + rules text
```

That hybrid-retrieval shape — structured filtering AND embeddings, plus a rules/
constraint layer on top — is what "RAG in production" actually looks like.

## Stack (all local)

| Concern    | Tool |
|------------|------|
| Python env & deps | [`uv`](https://github.com/astral-sh/uv) (project-managed Python 3.12) |
| Local models | [Ollama](https://ollama.com) |
| Embeddings | `nomic-embed-text` (768-dim) |
| Generation LLM | `qwen2.5:7b` |
| Vector math | `numpy` (hand-rolled first, then FAISS) |
| Card model | `pydantic` |

## Setup

Prerequisites: [`uv`](https://github.com/astral-sh/uv) and [Ollama](https://ollama.com) installed.

```bash
# Pull the local models (one-time)
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# Install Python deps into a project-local virtual environment
uv sync
```

## Roadmap

- [x] **Phase 0** — Toolchain & environment (uv, Ollama, models, verified pipeline)
- [x] **Phase 1** — Data layer: 4,285 cards normalized into validated `Card` objects, with the structured-metadata / free-text split that drives hybrid retrieval
- [ ] **Phase 2** — Core RAG (CLI): chunk → embed → store → retrieve → generate, by hand
- [ ] **Phase 3** — Swap hand-rolled search for a FAISS vector index
- [ ] **Phase 4** — Hybrid retrieval: metadata filtering + reranking
- [ ] **Phase 5** — Deck-building logic: format rules engine + "build me a deck"
- [ ] **Phase 6** — Evaluation: an eval set + retrieval/answer metrics
- [ ] **Phase 7** — UI + portfolio polish

## Data

Card data comes from the community-maintained open dataset
[`the-fab-cube/flesh-and-blood-cards`](https://github.com/the-fab-cube/flesh-and-blood-cards).
Raw downloads and generated indexes live under `data/` and are git-ignored; a script
fetches/builds them so the repo stays lean.

## License

Personal learning/portfolio project. *Flesh and Blood* is a trademark of Legend Story
Studios; card data belongs to its respective owners.
