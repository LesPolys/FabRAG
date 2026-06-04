"""Generation — Phase 3, the 'G' in RAG.

Retrieval (retrieval.py) finds the *right cards*. Generation turns them into a
*natural-language answer*. We run a local instruct model (qwen2.5:7b) via Ollama
and feed it the retrieved cards as grounding context.

The single most important idea here is **grounding**: the model must answer from
the cards we hand it, not from its own (often wrong, often outdated) memory of
Flesh and Blood. The system prompt enforces that contract — answer from the
provided cards, cite them by name, and admit when they don't cover the question.
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

SYSTEM_PROMPT = """\
You are FabRAG, an expert assistant for the Flesh and Blood (FAB) trading card game.

You will be given a CARDS section containing real card data retrieved from the
game's database, followed by a question. Follow these rules without exception:

1. Answer ONLY using the information in the CARDS section. Do not rely on prior
   knowledge of Flesh and Blood — card text and game rules change, and the CARDS
   section is the single source of truth.
2. If the CARDS section does not contain enough information to answer, say so
   plainly (e.g. "The retrieved cards don't cover that"). Never invent cards,
   rules, stats, or card names.
3. Cite the specific cards you use by their exact name.
4. Be concise and concrete. Prefer naming relevant cards and explaining why they
   fit over general game-theory filler.
"""


def _user_prompt(question: str, context: str) -> str:
    """Assemble the grounding context + the question into one user turn."""
    return f"CARDS:\n{context}\n\nQUESTION: {question}"


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
        options={"temperature": temperature},
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
        options={"temperature": temperature},
        stream=True,
    )
    for chunk in stream:
        piece = chunk.message.content
        if piece:
            yield piece
