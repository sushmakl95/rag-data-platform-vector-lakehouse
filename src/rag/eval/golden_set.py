"""Golden-set loader + per-query eval runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.rag.eval.metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    exact_match,
    faithfulness,
)


@dataclass(frozen=True)
class GoldenCase:
    question: str
    ground_truth: str
    must_contain_sources: list[str]


def load_golden(path: Path | str) -> list[GoldenCase]:
    doc = yaml.safe_load(Path(path).read_text())
    return [
        GoldenCase(
            question=c["question"],
            ground_truth=c["ground_truth"],
            must_contain_sources=c.get("must_contain_sources", []),
        )
        for c in doc["cases"]
    ]


def evaluate_case(
    case: GoldenCase,
    *,
    answer: str,
    retrieved_texts: list[str],
    ground_truth_snippets: list[str] | None = None,
) -> dict:
    gt_snips = ground_truth_snippets or [case.ground_truth]
    return {
        "query": case.question,
        "context_precision": context_precision(retrieved_texts, gt_snips),
        "context_recall": context_recall(retrieved_texts, gt_snips),
        "answer_relevance": answer_relevance(answer, case.question),
        "faithfulness": faithfulness(answer, retrieved_texts),
        "exact_match": exact_match(answer, case.ground_truth),
    }
