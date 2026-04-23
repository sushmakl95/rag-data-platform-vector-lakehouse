from __future__ import annotations

from pathlib import Path

from src.rag.eval.golden_set import evaluate_case, load_golden

REPO = Path(__file__).resolve().parents[2]


def test_load_golden_set():
    cases = load_golden(REPO / "configs" / "golden_set.yaml")
    assert len(cases) >= 3
    assert all(c.question and c.ground_truth for c in cases)


def test_evaluate_case_returns_all_metrics():
    cases = load_golden(REPO / "configs" / "golden_set.yaml")
    result = evaluate_case(
        cases[0],
        answer="Bronze is retained for 90 days via lifecycle",
        retrieved_texts=["Raw bronze data is retained for 90 days then expired by S3 lifecycle."],
    )
    assert set(result.keys()) >= {
        "query",
        "context_precision",
        "context_recall",
        "answer_relevance",
        "faithfulness",
        "exact_match",
    }
    assert 0 <= result["context_precision"] <= 1
