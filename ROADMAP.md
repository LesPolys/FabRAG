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

## ✅ Phase 4 — Rules-text corpus & multi-source retrieval
*Chunking + heterogeneous-corpus RAG — the most common real-world RAG pattern.*

- **Source** (`scripts/fetch_rules.py`): the CR as semantic HTML from
  rules.fabtcg.com (every rule paragraph carries its citable id) + the PDF
  kept as a future extraction-quality comparison. *(Decision: HTML primary.)*
- **Chunking** (`rules.py`): structure-aware — rule+subrules+examples form
  atomic groups packed into per-section chunks; oversized groups split at
  subrule boundaries with the parent rule repeated (surgical overlap). 581
  chunks across three sources (CR rules / CR glossary / keyword.json).
- **`Document` protocol** (`retrieval.py`): structural typing reduces
  "retrievable" to `doc_id` + `text_for_embedding`; card-specific filtering
  moved up into `scope()`; per-source cached indexes merged at runtime.
- **Mixed results**: measured the heterogeneous-corpus trap (cards score
  ~0.03 hotter than rule chunks → the right rule ranked 18th), fixed with
  per-source top-k quotas. `--source cards|rules|all` on both commands;
  `fabrag ask` cites rules with deep links.
- **Carried to Phase 5**: tune chunk size + quota split with the eval
  harness; observed one confabulated rule-number citation (true content,
  wrong attribution) — exactly what the grounding eval must catch.

## ✅ Phase 5 — Evaluation harness *(and the retrieval overhaul it forced)*
*Measure RAG quality before building on it — the measuring immediately paid off.*

- **Gold set** (`data/gold/gold_set.json`, 22 cases): ground truth from exact
  text search / CR lookup, never semantic search; expectations bind to stable
  identities (card names, CR rule numbers) so re-chunking can't invalidate
  them; `match: all|any` distinguishes "all relevant" from "alternatives".
- **Retrieval metrics** (`evaluation.py`): recall@k, MRR, nDCG@k; two lenses
  (`all` = production pipeline w/ quotas, `per-kind` = clean ranking quality).
- **The headline finding**: demos lied. Baseline cards recall@8 was **0.042**
  (dense nomic) — hub cards + weak discrimination on compositional queries —
  while rules retrieval was already perfect (1.000/0.964 MRR).
- **Eval-driven fixes, measured**: hand-rolled BM25 (`lexical.py`) + RRF
  fusion in the Retriever (`mode=hybrid|dense|lexical`), and an embedding
  model registry — `mxbai-embed-large` replaced nomic as default.
  **Cards recall@8: 0.042 → 0.625; overall: 0.652 → 0.864.** Pure BM25 still
  edges card recall (0.750) but loses MRR, rules, and paraphrase ability.
- **Grounding eval** (`grounding.py`): deterministic citation audit (every
  cited rule number/card must exist in the retrieved context) + qwen judge.
  First run caught a confabulated "CR 8.5.10" that the LLM judge graded
  "supported" — the layering argument in one example.
- **`fabrag eval`**: the whole suite behind one command.
- **Carried forward**: gold set should keep growing (22 → 40, esp. hard card
  cases); chunk-size and quota-split tuning now measurable but not yet swept;
  known-hard case: co-occurrence-structure queries ("destroy equipment on
  hit") defeat both scorers.

## 🔜 Phase 6 — Deck generation *(the headline)*
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
