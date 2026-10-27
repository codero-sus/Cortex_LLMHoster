from __future__ import annotations

from cortex_llmhoster.auth import (
    SESSION_TTL_SECONDS,
    create_session_token,
    is_authenticated,
    session_signing_key,
    valid_session_token,
    verify_admin_credentials,
)


def test_session_token_is_expiring_signed_and_revocable_by_key_rotation() -> None:
    now = 1_800_000_000
    token = create_session_token("first-secret", now=now)

    assert valid_session_token(token, "first-secret", now=now)
    assert valid_session_token(token, "first-secret", now=now + SESSION_TTL_SECONDS - 1)
    assert not valid_session_token(token, "first-secret", now=now + SESSION_TTL_SECONDS)
    assert not valid_session_token(token, "rotated-secret", now=now)
    assert not valid_session_token(f"{token}x", "first-secret", now=now)


def test_session_validation_rejects_malformed_and_non_ascii_tokens() -> None:
    assert not valid_session_token(None, "secret", now=100)
    assert not valid_session_token("not-a-token", "secret", now=100)
    assert not valid_session_token("١.0000", "secret", now=100)
    assert not valid_session_token("1." + "z" * 64, "secret", now=100)


def test_authentication_accepts_bearer_key_or_session_cookie() -> None:
    token = create_session_token("secret")

    assert is_authenticated("secret", "Bearer secret", token)
    assert is_authenticated("secret", "", token)
    assert is_authenticated("secret", "bEaReR secret", None)
    assert not is_authenticated("secret", "Bearer incorrect", None)
    assert not is_authenticated(None, "Bearer secret", token)


def test_account_changes_revoke_dashboard_sessions_but_not_bearer_api_keys() -> None:
    first_session_key = session_signing_key("api-key", "admin", "first-password")
    token = create_session_token(first_session_key)
    rotated_session_key = session_signing_key("api-key", "admin", "second-password")

    assert is_authenticated(
        "api-key", "", token, session_secret=first_session_key
    )
    assert not is_authenticated(
        "api-key", "", token, session_secret=rotated_session_key
    )
    assert is_authenticated(
        "api-key", "Bearer api-key", token, session_secret=rotated_session_key
    )


def test_admin_credentials_are_verified_as_a_pair() -> None:
    assert verify_admin_credentials("admin", "correct horse", "admin", "correct horse")
    assert not verify_admin_credentials("other", "correct horse", "admin", "correct horse")
    assert not verify_admin_credentials("admin", "wrong", "admin", "correct horse")
    assert not verify_admin_credentials(None, "correct horse", "admin", "correct horse")
    assert not verify_admin_credentials("admin", "correct horse", None, None)
    assert verify_admin_credentials("管理员", "pässwörd", "管理员", "pässwörd")
