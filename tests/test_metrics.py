from __future__ import annotations

from cortex_llmhoster.dashboard import InferenceMetrics


def test_metrics_reset_discards_completion_from_an_older_window() -> None:
    metrics = InferenceMetrics()
    old_request = metrics.start("old-model")

    metrics.reset()
    metrics.finish(old_request, 200)

    snapshot = metrics.snapshot()
    assert snapshot["in_flight"] == 0
    assert snapshot["total_requests"] == 0
    assert snapshot["completed_requests"] == 0
    assert snapshot["status_codes"] == {}
    assert snapshot["requests_by_model"] == {}


def test_metrics_count_client_and_server_errors_as_failures() -> None:
    metrics = InferenceMetrics()
    request = metrics.start("text-model")

    metrics.finish(request, 422)

    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 1
    assert snapshot["completed_requests"] == 1
    assert snapshot["failed_requests"] == 1
    assert snapshot["status_codes"] == {"422": 1}
    assert snapshot["requests_by_model"] == {"text-model": 1}
