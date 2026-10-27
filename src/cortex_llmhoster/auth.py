"""Signed, persistent browser sessions for the local dashboard."""

from __future__ import annotations

import hashlib
import hmac
import time

SESSION_COOKIE_NAME = "cortex_session"
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60
_SESSION_CONTEXT = b"cortex-dashboard-session:v1:"


def verify_admin_credentials(
    supplied_username: str | None,
    supplied_password: str | None,
    configured_username: str | None,
    configured_password: str | None,
) -> bool:
    """Compare one configured local-admin credential pair without leaking which part matched."""

    if (
        not isinstance(supplied_username, str)
        or not isinstance(supplied_password, str)
        or not isinstance(configured_username, str)
        or not isinstance(configured_password, str)
    ):
        return False
    username_matches = hmac.compare_digest(
        supplied_username.encode("utf-8"), configured_username.encode("utf-8")
    )
    password_matches = hmac.compare_digest(
        supplied_password.encode("utf-8"), configured_password.encode("utf-8")
    )
    return username_matches & password_matches


def session_signing_key(
    api_key: str | None,
    admin_username: str | None = None,
    admin_password: str | None = None,
) -> str | None:
    """Bind sessions to the required API key; account changes revoke prior sessions."""

    if api_key is None:
        return None
    master_secret = api_key
    if admin_username is None or admin_password is None:
        return master_secret
    account_context = (
        _SESSION_CONTEXT
        + b"account:"
        + admin_username.encode("utf-8")
        + b"\0"
        + admin_password.encode("utf-8")
    )
    return hmac.new(
        master_secret.encode("utf-8"), account_context, hashlib.sha256
    ).hexdigest()


def create_session_token(session_secret: str, *, now: int | None = None) -> str:
    """Create a 30-day signature token without storing credentials in the cookie."""

    expires = (int(time.time()) if now is None else now) + SESSION_TTL_SECONDS
    payload = f"{expires}".encode("ascii")
    signature = hmac.new(
        session_secret.encode("utf-8"), _SESSION_CONTEXT + payload, hashlib.sha256
    ).hexdigest()
    return f"{expires}.{signature}"


def valid_session_token(
    token: str | None,
    session_secret: str,
    *,
    now: int | None = None,
) -> bool:
    """Verify expiry and signature in constant time; changing auth secrets revokes sessions."""

    if not token:
        return False
    if len(token) > 128:
        return False
    try:
        expires_text, supplied_signature = token.split(".", maxsplit=1)
        if (
            not expires_text.isascii()
            or not expires_text.isdigit()
            or len(supplied_signature) != 64
            or any(character not in "0123456789abcdef" for character in supplied_signature)
        ):
            return False
        expires = int(expires_text)
    except (ValueError, AttributeError):
        return False
    current_time = int(time.time()) if now is None else now
    if expires <= current_time or expires > current_time + SESSION_TTL_SECONDS + 60:
        return False
    expected_signature = hmac.new(
        session_secret.encode("utf-8"),
        _SESSION_CONTEXT + expires_text.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied_signature, expected_signature)


def is_authenticated(
    api_key: str | None,
    authorization: str,
    session_token: str | None,
    *,
    session_secret: str | None = None,
) -> bool:
    """Accept the API's bearer key or a dashboard session cookie."""

    if api_key is not None:
        scheme, separator, supplied = authorization.partition(" ")
        if (
            separator
            and scheme.lower() == "bearer"
            and hmac.compare_digest(supplied.encode("utf-8"), api_key.encode("utf-8"))
        ):
            return True
    effective_session_secret = session_secret or api_key
    if effective_session_secret is None:
        return False
    return valid_session_token(session_token, effective_session_secret)
