from __future__ import annotations

import orjson

from cortex_llmhoster.dashboard import InferenceMetrics, InferenceResponseTelemetry


def test_metrics_reset_discards_completion_and_usage_from_an_older_window() -> None:
    metrics = InferenceMetrics()
    old_request = metrics.start("old-model")
    metrics.observe_first_token(old_request)
    metrics.observe_usage(old_request, prompt_tokens=12, completion_tokens=4)

    metrics.reset()
    metrics.finish(old_request, 200)

    snapshot = metrics.snapshot()
    assert snapshot["in_flight"] == 0
    assert snapshot["total_requests"] == 0
    assert snapshot["completed_requests"] == 0
    assert snapshot["status_codes"] == {}
    assert snapshot["requests_by_model"] == {}
    assert snapshot["generation_metrics"]["usage_requests"] == 0  # type: ignore[index]
    assert snapshot["generation_metrics_by_model"] == {}


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
    assert snapshot["generation_metrics"]["usage_requests"] == 0  # type: ignore[index]


def test_unmetered_success_does_not_create_empty_per_model_statistics() -> None:
    metrics = InferenceMetrics()
    request = metrics.start("unmetered-model")

    metrics.finish(request, 200)

    assert metrics.snapshot()["generation_metrics_by_model"] == {}


def test_generation_metrics_track_first_token_latency_and_weighted_throughput() -> None:
    metrics = InferenceMetrics()
    request = metrics.start("native-model")
    started_at = request.started_at
    metrics.observe_first_token(request, at=started_at + 0.125)
    metrics.observe_usage(request, prompt_tokens=40, completion_tokens=20)
    metrics.finish(request, 200, at=started_at + 0.625)

    snapshot = metrics.snapshot()
    generation = snapshot["generation_metrics"]
    model = snapshot["generation_metrics_by_model"]["native-model"]
    assert generation["usage_requests"] == 1
    assert generation["prompt_tokens"] == 40
    assert generation["completion_tokens"] == 20
    assert generation["first_token_samples"] == 1
    assert generation["average_first_token_latency_ms"] == 125.0
    assert generation["throughput_samples"] == 1
    assert generation["completion_tokens_per_second"] == 40.0
    assert model == generation


def test_json_response_usage_is_observed_without_retaining_response_text() -> None:
    metrics = InferenceMetrics()
    request = metrics.start("json-model")
    telemetry = InferenceResponseTelemetry(metrics, request, "application/json")
    response = orjson.dumps(
        {
            "choices": [{"message": {"content": "private generated response"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5},
        }
    )
    telemetry.feed(response[:20])
    telemetry.feed(response[20:])
    telemetry.finish(200)
    metrics.finish(request, 200)

    generation = metrics.snapshot()["generation_metrics"]
    assert generation["usage_requests"] == 1
    assert generation["prompt_tokens"] == 7
    assert generation["completion_tokens"] == 5
    assert generation["first_token_samples"] == 0


def test_sse_usage_and_first_text_delta_are_parsed_across_chunk_boundaries() -> None:
    metrics = InferenceMetrics()
    request = metrics.start("stream-model")
    telemetry = InferenceResponseTelemetry(metrics, request, "text/event-stream; charset=utf-8")
    frames = (
        b": heartbeat\r\n\r\n"
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n'
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\r\n\r\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3}}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )
    for offset in range(0, len(frames), 7):
        telemetry.feed(frames[offset : offset + 7])
    telemetry.finish(200)
    metrics.finish(request, 200)

    generation = metrics.snapshot()["generation_metrics"]
    assert generation["usage_requests"] == 1
    assert generation["prompt_tokens"] == 11
    assert generation["completion_tokens"] == 3
    assert generation["first_token_samples"] == 1
    assert generation["average_first_token_latency_ms"] >= 0
    assert generation["throughput_samples"] == 1
    assert generation["completion_tokens_per_second"] > 0


def test_responses_api_input_output_usage_aliases_are_supported() -> None:
    metrics = InferenceMetrics()
    request = metrics.start("responses-model")
    telemetry = InferenceResponseTelemetry(metrics, request, "text/event-stream")
    telemetry.feed(
        b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
        b'data: {"type":"response.completed","response":{"usage":{"input_tokens":8,"output_tokens":2}}}\n\n'
    )
    telemetry.finish(200)
    metrics.finish(request, 200)

    generation = metrics.snapshot()["generation_metrics"]
    assert generation["prompt_tokens"] == 8
    assert generation["completion_tokens"] == 2
    assert generation["first_token_samples"] == 1
