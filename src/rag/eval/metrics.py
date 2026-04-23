"""RAGAS-style eval metrics, all pure-Python + testable.

- context_precision: fraction of retrieved chunks that overlap with ground-truth snippets
- context_recall: fraction of ground-truth snippets covered by retrieved chunks
- answer_relevance: token-overlap between answer and question
- faithfulness: token-overlap between answer and retrieved contexts (no invented facts)
- exact_match: answer == ground truth (case / whitespace insensitive)

In production you'd swap these for LLM-judge versions; the token-overlap implementations
here give a reproducible, zero-cost baseline.
"""

from __future__ import annotations

import re


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}


def context_precision(retrieved: list[str], ground_truth_snippets: list[str]) -> float:
    if not retrieved:
        return 0.0
    gt_tokens = (
        set().union(*(_tokens(s) for s in ground_truth_snippets))
        if ground_truth_snippets
        else set()
    )
    if not gt_tokens:
        return 0.0
    hits = sum(1 for r in retrieved if _tokens(r) & gt_tokens)
    return hits / len(retrieved)


def context_recall(retrieved: list[str], ground_truth_snippets: list[str]) -> float:
    if not ground_truth_snippets:
        return 0.0
    retrieved_tokens = set().union(*(_tokens(r) for r in retrieved)) if retrieved else set()
    if not retrieved_tokens:
        return 0.0
    hits = sum(1 for g in ground_truth_snippets if _tokens(g) & retrieved_tokens)
    return hits / len(ground_truth_snippets)


def answer_relevance(answer: str, question: str) -> float:
    q, a = _tokens(question), _tokens(answer)
    if not q:
        return 0.0
    return len(q & a) / len(q)


def faithfulness(answer: str, contexts: list[str]) -> float:
    ans = _tokens(answer)
    if not ans:
        return 0.0
    ctx = set().union(*(_tokens(c) for c in contexts)) if contexts else set()
    if not ctx:
        return 0.0
    return len(ans & ctx) / len(ans)


def exact_match(answer: str, ground_truth: str) -> float:
    norm_a = re.sub(r"\s+", " ", answer.strip().lower())
    norm_g = re.sub(r"\s+", " ", ground_truth.strip().lower())
    return 1.0 if norm_a == norm_g else 0.0


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {"count": 0}
    keys = {k for r in results for k in r if k != "query"}
    return {
        "count": len(results),
        **{k: sum(r.get(k, 0.0) for r in results) / len(results) for k in keys},
    }
