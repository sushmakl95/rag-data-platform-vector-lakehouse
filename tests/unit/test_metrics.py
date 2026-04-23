from __future__ import annotations

from src.rag.eval.metrics import (
    aggregate,
    answer_relevance,
    context_precision,
    context_recall,
    exact_match,
    faithfulness,
)


def test_context_precision_full_overlap():
    retrieved = ["delta lake supports time travel"]
    gt = ["delta lake supports time travel via log"]
    assert context_precision(retrieved, gt) == 1.0


def test_context_precision_no_overlap():
    assert context_precision(["xyzzy"], ["delta lake"]) == 0.0


def test_context_recall():
    retrieved = ["delta lake architecture"]
    gt = ["delta lake", "marketing campaigns"]
    recall = context_recall(retrieved, gt)
    assert 0 < recall < 1


def test_answer_relevance():
    assert answer_relevance("the delta lake format", "what is delta lake?") > 0.5


def test_faithfulness():
    assert faithfulness("delta lake", ["delta lake is a table format"]) == 1.0
    assert faithfulness("martian invasion", ["delta lake is a table format"]) == 0.0


def test_exact_match_normalised():
    assert exact_match("  Hello,  World  ", "hello, world") == 1.0
    assert exact_match("hello", "world") == 0.0


def test_aggregate_averages_metrics():
    agg = aggregate(
        [
            {"query": "q1", "m1": 0.8, "m2": 0.5},
            {"query": "q2", "m1": 0.6, "m2": 0.7},
        ]
    )
    assert agg["count"] == 2
    assert abs(agg["m1"] - 0.7) < 1e-6
    assert abs(agg["m2"] - 0.6) < 1e-6
