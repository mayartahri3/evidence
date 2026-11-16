import pytest

from pipelines.evaluation.evaluate_retrieval import compute_retrieval_metrics


def test_retrieval_metrics_use_relevant_ranks_within_top_k():
    results = [
        {"paper_id": "irrelevant"},
        {"paper_id": "relevant-1"},
        {"paper_id": "relevant-2"},
    ]

    metrics = compute_retrieval_metrics(
        results, {"relevant-1", "relevant-2", "not-retrieved"}, top_k=2
    )

    assert metrics["recall_at_k"] == pytest.approx(1 / 3)
    assert metrics["mrr_at_k"] == pytest.approx(1 / 2)


def test_retrieval_metrics_return_zero_when_no_relevant_result_is_retrieved():
    metrics = compute_retrieval_metrics(
        [{"paper_id": "unrelated"}], {"relevant"}, top_k=5
    )

    assert metrics == {"recall_at_k": 0.0, "mrr_at_k": 0.0}


def test_retrieval_metrics_do_not_count_duplicate_relevant_results_twice():
    metrics = compute_retrieval_metrics(
        [{"paper_id": "relevant"}, {"paper_id": "relevant"}],
        {"relevant", "not-retrieved"},
        top_k=2,
    )

    assert metrics["recall_at_k"] == pytest.approx(1 / 2)
    assert metrics["mrr_at_k"] == pytest.approx(1.0)


def test_retrieval_metrics_reject_invalid_inputs():
    with pytest.raises(ValueError, match="top_k"):
        compute_retrieval_metrics([], {"relevant"}, top_k=0)
    with pytest.raises(ValueError, match="relevant"):
        compute_retrieval_metrics([], set(), top_k=5)
