"""Small helpers for validating HTTP-level framing metadata."""

from __future__ import annotations

from starlette.datastructures import Headers


def parse_content_length(value: str | None) -> int | None:
    """Parse a decimal-only Content-Length, rejecting ambiguous or invalid forms."""

    if value is None:
        return None
    normalized = value.strip(" \t")
    if not normalized.isascii() or not normalized.isdecimal():
        raise ValueError("Content-Length must contain ASCII decimal digits.")
    return int(normalized)


def content_length_from_headers(headers: Headers) -> int | None:
    """Reject duplicate Content-Length fields instead of trusting the first one."""

    values = headers.getlist("content-length")
    if len(values) > 1:
        raise ValueError("Content-Length must appear at most once.")
    return parse_content_length(values[0] if values else None)
