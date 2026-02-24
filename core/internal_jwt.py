from __future__ import annotations

from jose import JWTError, jwt

from config import (
    INTERNAL_JWT_ALGORITHM,
    INTERNAL_JWT_AUDIENCE,
    INTERNAL_JWT_ISSUER,
    INTERNAL_JWT_SECRET,
)


def extract_bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None
    value = authorization_header.strip()
    if not value:
        return None
    if not value.lower().startswith("bearer "):
        return None
    token = value.split(" ", 1)[1].strip()
    return token or None


def verify_internal_jwt(token: str) -> dict:
    if not INTERNAL_JWT_SECRET:
        raise ValueError("INTERNAL_JWT_SECRET is not configured")

    return jwt.decode(
        token,
        INTERNAL_JWT_SECRET,
        algorithms=[INTERNAL_JWT_ALGORITHM],
        issuer=INTERNAL_JWT_ISSUER,
        audience=INTERNAL_JWT_AUDIENCE,
    )


def validate_internal_authorization_header(authorization_header: str | None) -> tuple[bool, str]:
    token = extract_bearer_token(authorization_header)
    if not token:
        return False, "Missing internal Authorization bearer token"

    try:
        payload = verify_internal_jwt(token)
    except (JWTError, ValueError):
        return False, "Invalid internal token"

    if payload.get("typ") != "internal":
        return False, "Invalid internal token type"

    return True, "ok"
