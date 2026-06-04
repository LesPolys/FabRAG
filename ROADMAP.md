# FabRAG Roadmap

A fully-local, RAG-based AI deck-building assistant for the Flesh and Blood (FAB)
trading card game. Twofold purpose: learn RAG/AI fundamentals from the internals
up, and learn FAB along the way. No cloud APIs — Ollama for the LLM and
embeddings, everything runs on the local machine.

**Workflow:** one branch per phase (`phase-N-...`), merged to `main` with a
`--no-ff` merge commit when complete. This document is living — update it as
phases land.

**Status legend:** ✅ done · 🔜 next · ⬜ planned

---

## ✅ Phase 0 — Scaffold & local toolchain
Project skeleton, packaging (`uv` + hatchling), and the local AI stack (Ollama
with `nomic-embed-text` for embeddings and `qwen2.5:7b` for generation).

## ✅ Phase 1 — Data layer
Typed, validated `Card` model over the raw FAB card JSON. The central
architectural split is made explicit here: **structured metadata → filtering**,
**free text → embedding**. Full six-format `Legality` (CC, Blitz, Commoner,
Living Legend, Silver Age, UPF), stored losslessly.

## ✅ Phase 2 — Core RAG (retrieval)
- **Embedding layer** (`embeddings.py`): text → unit-normalized 768-dim vectors
  via `nomic-embed-text`, with the required `search_document:` / `search_query:`
  task prefixes baked in.
- **Hybrid retrieval** (`retrieval.py`): filter-then-rank. A `CardFilter`
  (color/pitch/class/talent/legality/…) shrinks the corpus, then cosine
  similarity (a single matrix multiply, since vectors are normalized) ranks the
  survivors. Index caches to a fingerprinted `.npz` (~12.5 MB; ~40s build →
  ~0.2s reload).

## ✅ Phase 3 — End-to-end RAG assistant
- **Generation** (`generation.py`): grounded answers via `qwen2.5:7b` — answers
  only from supplied cards, cites them, refuses when they don't cover the
  question.
- **Orchestration** (`rag.py`): retrieve → format context → generate.
- **CLI** (`cli.py`): `fabrag search` (ranked cards) and `fabrag ask` (streamed
  grounded answer).
- **Deck rules** (`deck.py`): eligibility (the class/talent **subset rule** +
  format legality) and deck validation (min size, max-3 copies). Wired into
  retrieval via a `predicate` hook and the CLI `--hero` flag, so results are
  restricted to a hero's legal pool.

---

## 🔜 Phase 4 — Rules-text corpus & multi-source retrieval
*Goal: chunking + heterogeneous-corpus RAG — the most common real-world RAG
pattern, and one cards let us skip (each card is conveniently one document).*

1. **Source** the FAB rules text (comprehensive rulebook + keyword glossary)
   into `data/raw/`. *(Decision: which official/permitted source.)*
2. **Chunk** the prose by section/heading with size limits + overlap; learn why
   chunk size is a real tuning knob.
3. **Generalize the index** — a `Document` abstraction so `Retriever` embeds both
   `Card`s and `RuleChunk`s. Refactors `retrieval.py` from Card-specific to
   source-agnostic (the largest structural change in the plan).
4. **Mixed results** — retrieval returns cards *and* rules; context formatting
   renders each by type; `fabrag ask` can cite a rule. Add `--source
   cards|rules|all`.

## ⬜ Phase 5 — Evaluation harness
*Goal: measure RAG quality — retrieval metrics + answer grounding — before
building the hardest component, so it guards against regressions.*

1. Hand-author a **gold set** (~20–40 cases: query → expected card/rule).
2. **Retrieval metrics**: recall@k, MRR/nDCG.
3. **Grounding eval**: does the answer cite only retrieved cards? (heuristic +
   local-LLM-as-judge).
4. `fabrag eval` → a report with baseline numbers; use it to tune k, chunk size,
   prefixes.

## ⬜ Phase 6 — Deck generation *(the headline)*
*Goal: constraint-guided generation / a lightweight agentic loop.*

1. **Input**: hero + optional strategy/archetype + format.
2. **Candidate sourcing**: hero-pool retrieval driven by strategy-derived queries.
3. **Assembly + repair loop**: LLM proposes a list → `validate_deck()` checks →
   errors fed back → re-propose until legal. *The core agentic lesson.*
4. **Curve/ratio heuristics**: pitch balance, copy counts, loadout slots.
5. **Explanation**: grounded rationale. → `fabrag build --hero X --strategy
   "aggro arcane"` yields a guaranteed-legal decklist + why.

## ⬜ Phase 7 — Card-display UI *(capstone demo)*
*Goal: surface RAG results visually; a shareable artifact.*

1. **Stack choice** *(decision)*: fast win (Streamlit/Gradio) vs. learn-internals
   (FastAPI + minimal JS).
2. Search view (card grid with images via `image_url`), Ask view (answer + cited
   card images), Deck view (generated list rendered by pitch/type).
3. Thin layer over existing `rag`/`deck` modules — no logic duplication.

---

### Sequencing rationale
4 generalizes retrieval (cleaner before more is built on it) and enriches
deck-gen with rules knowledge → 5 baselines the unified system → 6 is measured
and guarded by 5 → 7 visualizes everything last.
