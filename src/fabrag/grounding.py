"""Grounding evaluation — Phase 5's second half: is the ANSWER honest?

Retrieval metrics (evaluation.py) ask "did the right documents reach the
context?" This module asks the next question: "did the generated answer stay
inside that context?" The failure mode is specific and we've already caught it
in the wild: during Phase 4 the model answered a dominate question correctly
but attributed it to "CR 7.3.2" — a rule that was NOT in its context. True
content, fabricated citation. An answer you can't verify is an answer you
can't trust, which defeats the point of RAG.

Two complementary checks:

  1. CITATION HEURISTIC (deterministic, fast, narrow): extract every rule
     number and card name the answer mentions, and verify each one actually
     appeared in the retrieved context. Catches fabricated attributions with
     zero false negatives for rule numbers (they're regular enough to parse).
     Card names are fuzzier — we scan the answer for ALL corpus card names, so
     we can also catch the model name-dropping cards it was never shown.

  2. LLM-AS-JUDGE (broad, soft): a second local-model call that grades whether
     the answer's CLAIMS are supported by the context — catching unsupported
     statements that cite nothing at all. A 7B judge is imperfect (it shares
     failure modes with the 7B answerer), so its verdicts are a signal to
     read, not a gate to enforce. Industry practice uses a stronger judge than
     answerer; fully-local means we accept this limitation and say so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import ollama

from .cards import Card
from .generation import CHAT_MODEL
from .retrieval import SearchResult
from .rules import RuleChunk

# --- citation extraction ------------------------------------------------------
# "CR 7.3.2", "rule 8.3.4a", "[7.2.3]" — the shapes the model produces in
# practice (we told it to cite "CR x.y.z", but graders shouldn't trust the
# answerer to follow instructions — that's the thing under test).
_RULE_REF = re.compile(r"\b(?:CR|rule)\s+(\d+(?:\.-?\d+)*[a-z]?)\b|\[(\d+(?:\.-?\d+)+[a-z]?)\]", re.I)
_TITLE_REF = re.compile(r"\b(?:CR\s+)?(?:Glossary|Keyword):\s*([A-Za-z][\w' -]*)", re.I)


def extract_rule_citations(answer: str) -> list[str]:
    """Every rule number the answer cites, normalized without the 'cr' prefix."""
    out = []
    for m in _RULE_REF.finditer(answer):
        out.append((m.group(1) or m.group(2)))
    return sorted(set(out))


def extract_title_citations(answer: str) -> list[str]:
    """Glossary/keyword names the answer cites, e.g. 'CR Glossary: Dominate'."""
    return sorted({m.group(1).strip().lower() for m in _TITLE_REF.finditer(answer)})


def context_rule_ids(results: list[SearchResult]) -> set[str]:
    """All rule numbers present in the retrieved context (without 'cr')."""
    ids: set[str] = set()
    for r in results:
        if isinstance(r.doc, RuleChunk):
            ids.update(rid.removeprefix("cr") for rid in r.doc.rule_ids)
    return ids


def find_card_mentions(answer: str, all_card_names: set[str]) -> set[str]:
    """Which corpus card names does the answer mention?

    Heuristic with a documented bias: multi-word names match case-insensitively
    ("sink below" is unambiguous), but single-word names must match with their
    exact capitalization — "Snatch" the card vs "snatch" the verb. This trades
    a few misses for far fewer false alarms.
    """
    found = set()
    lower = answer.lower()
    for name in all_card_names:
        if " " in name or "-" in name:
            if name.lower() in lower:
                found.add(name)
        elif re.search(rf"\b{re.escape(name)}\b", answer):
            found.add(name)
    return found


# --- the per-answer verdict ---------------------------------------------------
@dataclass(frozen=True)
class GroundingResult:
    question: str
    answer: str
    cited_rules: list[str]          # rule numbers the answer cites
    confabulated_rules: list[str]   # cited but NOT in the retrieved context
    cited_titles: list[str]         # glossary/keyword names cited
    confabulated_titles: list[str]
    mentioned_cards: list[str]      # corpus card names appearing in the answer
    ungrounded_cards: list[str]     # mentioned but NOT in the retrieved context
    judge_verdict: str | None = None        # supported / partial / unsupported
    judge_notes: list[str] = field(default_factory=list)

    @property
    def citations_clean(self) -> bool:
        return not (self.confabulated_rules or self.confabulated_titles or self.ungrounded_cards)


def check_citations(
    question: str,
    answer: str,
    results: list[SearchResult],
    all_card_names: set[str],
) -> GroundingResult:
    """The deterministic half: every citation must point into the context."""
    ctx_rules = context_rule_ids(results)
    ctx_titles = {
        (r.doc.title or "").lower() for r in results if isinstance(r.doc, RuleChunk)
    }
    ctx_cards = {r.doc.name for r in results if isinstance(r.doc, Card)}

    cited_rules = extract_rule_citations(answer)
    # A citation is grounded if the context contains that rule or anything
    # under it ("CR 7.3" is fine if 7.3.2 was retrieved — citing the section
    # that contains your evidence is generalization, not fabrication).
    confab_rules = [
        c for c in cited_rules
        if not any(rid == c or rid.startswith(c + ".") or rid.startswith(c) and rid[len(c):][:1].isalpha()
                   for rid in ctx_rules)
    ]

    cited_titles = extract_title_citations(answer)
    confab_titles = [t for t in cited_titles if t not in ctx_titles]

    mentioned = find_card_mentions(answer, all_card_names)
    ungrounded = sorted(mentioned - ctx_cards)

    return GroundingResult(
        question=question,
        answer=answer,
        cited_rules=cited_rules,
        confabulated_rules=confab_rules,
        cited_titles=cited_titles,
        confabulated_titles=confab_titles,
        mentioned_cards=sorted(mentioned),
        ungrounded_cards=ungrounded,
    )


# --- the LLM judge --------------------------------------------------------------
_JUDGE_PROMPT = """\
You are a strict fact-checking grader. You will be given a CONTEXT (the only \
permitted source of truth), a QUESTION, and an ANSWER. Judge whether every \
factual claim in the ANSWER is supported by the CONTEXT. Ignore style; judge \
only factual support. Respond with JSON only:
{"verdict": "supported" | "partial" | "unsupported", "unsupported_claims": ["..."]}\
"""


def judge_answer(question: str, context: str, answer: str, *, model: str = CHAT_MODEL) -> tuple[str, list[str]]:
    """Ask a local model to grade answer-vs-context support.

    Returns (verdict, unsupported_claims). Failures of the judge itself
    (malformed JSON, etc.) come back as verdict="judge_error" — never let a
    broken grader masquerade as a passing grade.
    """
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"},
        ],
        options={"temperature": 0.0},
        format="json",
    )
    try:
        data = json.loads(resp.message.content)
        verdict = str(data.get("verdict", "judge_error"))
        claims = [str(c) for c in data.get("unsupported_claims", [])]
        if verdict not in ("supported", "partial", "unsupported"):
            verdict = "judge_error"
        return verdict, claims
    except (json.JSONDecodeError, AttributeError, TypeError):
        return "judge_error", []


# --- suite runner ---------------------------------------------------------------
def grounding_eval(
    questions: list[str],
    *,
    k: int = 8,
    use_judge: bool = True,
) -> list[GroundingResult]:
    """Generate an answer per question through the production pipeline, then
    grade it. Slow (one or two LLM calls per question) — run on a handful of
    questions, not the whole gold set, unless you have coffee."""
    from .cards import load_cards
    from .rag import answer as rag_answer
    from .rag import format_context

    all_names = {c.name for c in load_cards()}
    out: list[GroundingResult] = []
    for q in questions:
        response = rag_answer(q, k=k)
        result = check_citations(q, response.answer, response.results, all_names)
        if use_judge and response.results:
            verdict, claims = judge_answer(q, format_context(response.results), response.answer)
            result = GroundingResult(
                **{**result.__dict__, "judge_verdict": verdict, "judge_notes": claims}
            )
        out.append(result)
    return out


def render_grounding(results: list[GroundingResult]) -> str:
    lines = [f"Grounding eval — {len(results)} answers", ""]
    clean = sum(1 for r in results if r.citations_clean)
    lines.append(f"citations clean: {clean}/{len(results)}")
    if any(r.judge_verdict for r in results):
        for v in ("supported", "partial", "unsupported", "judge_error"):
            n = sum(1 for r in results if r.judge_verdict == v)
            if n:
                lines.append(f"judge {v}: {n}")
    lines.append("")
    for r in results:
        flags = []
        if r.confabulated_rules:
            flags.append(f"CONFABULATED RULES {r.confabulated_rules}")
        if r.confabulated_titles:
            flags.append(f"CONFABULATED TITLES {r.confabulated_titles}")
        if r.ungrounded_cards:
            flags.append(f"UNGROUNDED CARDS {r.ungrounded_cards}")
        status = "; ".join(flags) if flags else "clean"
        verdict = f" | judge: {r.judge_verdict}" if r.judge_verdict else ""
        lines.append(f"  {status}{verdict}  — {r.question!r}")
        for claim in r.judge_notes[:3]:
            lines.append(f"      ! {claim}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick self-check on a few gold-set questions (full runs go via `fabrag eval`).
    from .evaluation import load_gold_set

    questions = [c.query for c in load_gold_set() if c.kind == "rule"][:4]
    print(render_grounding(grounding_eval(questions)))
