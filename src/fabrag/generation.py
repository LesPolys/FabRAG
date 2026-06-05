"""Generation — Phase 3, the 'G' in RAG.

Retrieval (retrieval.py) finds the *right documents* — cards and rules excerpts.
Generation turns them into a *natural-language answer*. We run a local instruct
model (qwen2.5:7b) via Ollama and feed it the retrieved context as grounding.

The single most important idea here is **grounding**: the model must answer from
the context we hand it, not from its own (often wrong, often outdated) memory of
Flesh and Blood. The system prompt enforces that contract — answer from the
provided cards/rules, cite them, and admit when they don't cover the question.
That discipline is exactly what separates RAG from "just ask a chatbot": the
answer is traceable to real corpus data.

This module is deliberately PURE LLM plumbing — it takes a question and an
already-formatted context string. Turning Card objects into that context string
is rag.py's job, keeping the retrieve/format/generate stages cleanly separable.
"""

from __future__ import annotations

from collections.abc import Iterator

import ollama

CHAT_MODEL = "qwen2.5:7b"

# Low temperature: this is grounded factual Q&A, not creative writing. We want
# the model to stick to the cards, not embellish.
DEFAULT_TEMPERATURE = 0.2

# Ollama's default context window (often 4096) TRUNCATES SILENTLY — with
# Phase 8's bigger rule chunks (up to ~4400 chars each), a k=8 mixed context
# could lose its tail without any error. 8k tokens covers the worst case.
NUM_CTX = 8192

SYSTEM_PROMPT = """\
You are FabRAG, an expert assistant for the Flesh and Blood (FAB) trading card game.

You will be given a CONTEXT section containing retrieved game data — card entries
(marked "Card:") and official rules excerpts (marked "Rule ..."), followed by a
question. Follow these rules without exception:

1. Answer ONLY using the information in the CONTEXT section. Do not rely on prior
   knowledge of Flesh and Blood — card text and game rules change, and the CONTEXT
   section is the single source of truth.
2. If the CONTEXT section does not contain enough information to answer, say so
   plainly (e.g. "The retrieved cards and rules don't cover that"). Never invent
   cards, rules, stats, or card names.
3. Cite your sources: cards by their exact name, rules by their rule number or
   glossary/keyword name EXACTLY as written in the CONTEXT (e.g. "CR 7.3.2",
   "CR Glossary: Dominate"). Never cite a rule number that does not appear in
   the CONTEXT.
4. Be concise and concrete. Prefer naming relevant cards and citing specific rules
   over general game-theory filler.
"""


def _user_prompt(question: str, context: str) -> str:
    """Assemble the grounding context + the question into one user turn."""
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"


def generate(
    question: str,
    context: str,
    *,
    model: str = CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Answer `question` grounded in `context`, returning the full reply text."""
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(question, context)},
        ],
        options={"temperature": temperature, "num_ctx": NUM_CTX},
    )
    return resp.message.content


def generate_stream(
    question: str,
    context: str,
    *,
    model: str = CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Iterator[str]:
    """Same as generate(), but yield the reply token-by-token for live CLI output."""
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(question, context)},
        ],
        options={"temperature": temperature, "num_ctx": NUM_CTX},
        stream=True,
    )
    for chunk in stream:
        piece = chunk.message.content
        if piece:
            yield piece
