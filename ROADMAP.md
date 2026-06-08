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

## ✅ Phase 6 — Deck generation *(the headline)*
*Constraint-guided generation: the generator + verifier + feedback loop.*

- **Pipeline** (`build.py`): strategy —LLM→ 4-6 retrieval queries —hybrid
  search over the hero-legal pool→ quoted candidate sheet —LLM→ JSON
  decklist → resolve + `validate_deck()` → errors fed back → re-propose
  (≤4 rounds) → **deterministic finisher** (drop/clamp/pad, every repair
  disclosed) makes "guaranteed legal" honest. Legality hard, grounding soft
  (off-sheet-but-legal picks warn, don't fail).
- **Heuristics**: pitch-balance targets guide the finisher's padding and
  produce quality warnings (oversize main deck, off-target pitch ratios).
- **`fabrag build --hero X --strategy "..." --format blitz`**: decklist
  grouped loadout-then-pitch + a game-plan explanation grounded in the
  decklist itself (reuses the fabrag-ask anti-hallucination contract).
- **Bugs the loop surfaced**: deck.py admitted Token/Macro/Landmark/
  Demi-Hero cards (the generator "drafted" the Quicken token — CR 1.3.2
  taxonomy now enforced); "best attempt = fewest errors" preferred an empty
  deck (1 error) over a fixable 78-card one (badness now counts deficit).
- **Observed**: legal in 2-4 LLM rounds; 7B round-to-round variance is large
  (68→154 cards), which is the finisher's reason to exist.
- **Carried forward**: equipment/loadout slot limits + Blitz exact-40 still
  unmodeled (deck.py known simplifications); strategy-query diversity is
  poor (hero name in every query); deck *quality* has no eval yet — only
  legality is measured.

## ✅ Phase 7 — Card-display UI *(capstone demo)*
*RAG results, visually — `fabrag serve` → http://127.0.0.1:8000.*

- **Stack** *(decision)*: FastAPI + framework-free JS/CSS — chosen for card
  display quality (true 450:628 aspect grid, hover-zoom, lightbox, lazy
  loading, `played_horizontally` cards rotated in their cells) and the
  learn-web-internals path.
- **Thin-skin rule held**: every route is parse → call-what-the-CLI-calls →
  serialize; no logic in the web layer. Two front doors, one engine.
- **Search view**: numbered pages over the full (500-cap) result set with
  server-side sorting (name/pitch/cost/power/defense; None-stats last) —
  sort-then-page, so page 1 of "cost ↑" is the cheapest matches overall.
  Rule hits render as citation chips with CR deep links.
- **Ask view**: SSE streaming (grounding event renders citations before the
  first token arrives, mirroring the CLI's results-then-stream shape).
- **Build view**: deck grouped loadout/pitch-1/2/3 with color accents and
  copy badges; finisher notes + quality warnings disclosed; game plan below.

---

*Phases 0–7 (the original plan) are complete: `fabrag search` · `ask` ·
`build` · `eval` · `serve`. Phases 8–10 extend the deck-building flow into a
real product surface.*

## ✅ Phase 8 — Eval deepening & retrieval tuning
*Turned the knobs Phase 5 built the dial for — one changed, one validated,
one experiment concluded.*

- **Gold set 22 → 38** (specializations, multi-constraint, broad any-of
  needs). Baseline at 38: overall 0.842 recall@8 / 0.723 MRR (per-kind).
- **Chunk-size sweep** (`scripts/sweep_chunks.py`): recall saturates at 1.0
  from 1000 chars; MRR/nDCG peak at the shipped **1500/3000 — the guess
  survived measurement** and is now a choice. The sweep's own first version
  evaluated a rules-only retriever and recommended 2200/4400; the merged-
  corpus (production) numbers reversed that — BM25 statistics differ.
  Measure the system, not a flattering simplification.
- **Quota sweep** (`scripts/sweep_quota.py`): **RULES_SHARE 0.30 → 0.40** —
  three rules slots at k=8 lift rules recall 0.850 → 0.950 and overall
  0.737 → 0.781 for a 0.018 card cost; a fourth slot bought nothing.
- **Reranker probe** (`scripts/probe_reranker.py`): an LLM judge rescues
  in-window misses brilliantly (Glacial Horns #13 → #1) but the truly hard
  cases sit at depth ~60-90+, where one-shot listwise judging fails (context
  + 7B attention limits). Verdict: a pointwise reranker (1 call/candidate)
  would work but is too slow for interactive local search — candidate-depth,
  not ranking, is the hard cases' bottleneck. Recorded, not built.
- Hygiene: explicit `num_ctx=8192` on generation calls (Ollama's default
  truncates silently).

## ✅ Phase 9 — Card-pool registration & sideboarding
*Modelled registration the way the game does — and the TRP read paid off the
same way Phase 5's eval did: it exposed a shipped bug.*

- **The headline finding — a per-NAME copy bug.** The TRP limits copies "per
  unique card", and CR 2.7.1/2.8.1 define uniqueness as **name + pitch**.
  deck.py counted by name, so it wrongly rejected a legal 3×red + 3×blue of one
  name (six cards, two unique cards, each ≤3) and couldn't express Blitz's
  1-per-unique rule. Fixed across the validator, the build prompt, and the
  finisher's clamp — now keyed on (name, pitch). A real Blitz build proves it:
  Ira's deck runs Flic Flak at pitch 1, 2 AND 3, one copy each, all legal.
- **`formats.py`** — one `FormatRules` table (size / pool cap / copy limit /
  hero age), replacing the scattered MIN_DECK_SIZE / MAX_COPIES / alias maps.
  CC: adult, ≥60, ≤80 pool, ≤3. Blitz: young, exactly-40, ≤52 pool, ≤1.
- **`pool.py` — `CardPool` + `validate_pool`**: hero + starting deck + equipped
  inventory + sideboard. Models the rules that only exist once registration
  does — ≤1 arena-card per Head/Chest/Arms/Legs/Off-Hand and ≤2 weapon-hands
  (CR 4.1.4a), the pool cap, whole-pool copy limits, and hero age. The two
  latent bugs the TRP read named (Blitz exact-40 + 1-copy, hero-age legality)
  are now enforced; `build` fails fast on an adult hero in Blitz.
- **`build.py` full registrations**: the picked equipment becomes the starting
  inventory (slot-clamped, auto-equips a weapon if none chosen); a new LLM call
  suggests a **matchup sideboard** (reason + swap target per card), grounded in
  the hero pool and capped to the format's pool rules — best-effort, never
  breaking the legal-registration contract. Hardened `_resolve_picks` against
  the 7B's bare-string picks (a latent crash the new path surfaced).
- **Rendered as three sections** (CLI + web): inventory by slot · deck by pitch
  · sideboard with rationale, plus the pool total / cap.
- Eval unchanged (0.842 recall@8 / 0.723 MRR) — the pool layer doesn't touch
  retrieval.

## ⬜ Phase 10 — Deck workspace *(chat + stats)*
*Goal: from "generate a deck" to "work on a deck".*

1. **Deck stats panel**: pitch distribution, cost curve, card-type breakdown
   (API + web UI).
2. **Hero card displayed** in the deck view, with the registration sections
   from Phase 9.
3. **Deck chat**: a conversational agent grounded in the current pool +
   rules corpus — discuss the deck, ask "why this card", request swap
   suggestions; the agent can search the hero pool and cite rules. Extends
   Phase 6's agentic loop with multi-turn state.

### Carryover parking lot
Strategy-query diversity in build.py; deck-QUALITY eval (legality is
measured, goodness isn't) — natural companion to Phase 10's stats.

---

### Sequencing rationale
4 generalizes retrieval (cleaner before more is built on it) and enriches
deck-gen with rules knowledge → 5 baselines the unified system → 6 is measured
and guarded by 5 → 7 visualizes everything last.
