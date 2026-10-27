"""Small JSON response helpers shared by the public API and dashboard."""

from __future__ import annotations

import orjson
from starlette.responses import Response


class OrjsonResponse(Response):
    """Render JSON with orjson for a small response-path overhead."""

    media_type = "application/json"

    def render(self, content: object) -> bytes:
        return orjson.dumps(content)


def error_response(
    message: str,
    *,
    status_code: int,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> OrjsonResponse:
    error: dict[str, object] = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    if param is not None:
        error["param"] = param
    return OrjsonResponse({"error": error}, status_code=status_code, headers=headers)
