from fastapi import Request
from fastapi.responses import JSONResponse

from config import (
    INTERNAL_JWT_ENABLED,
    MTLS_ENABLED,
    SECURITY_BLOCKED_USER_AGENT_KEYWORDS,
)

from core.internal_jwt import validate_internal_authorization_header

EXEMPT_PATHS = {"/docs", "/redoc", "/openapi.json"}


async def security_middleware(request: Request, call_next):
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    if MTLS_ENABLED:
        # mTLS is enforced at the transport layer by Uvicorn (ssl_cert_reqs=ssl.CERT_REQUIRED).
        # Any request that reaches this point over HTTPS already had its client certificate
        # verified during the TLS handshake — Uvicorn rejects invalid/missing certs before
        # the ASGI app is invoked.
        # Do NOT use X-Forwarded-Proto — it is attacker-controlled.
        # request.url.scheme is set by Uvicorn from the actual socket SSL state, not headers.
        if request.url.scheme != "https":
            return JSONResponse(
                status_code=403,
                content={"detail": "mTLS required: HTTPS connection required"},
            )

    if INTERNAL_JWT_ENABLED:
        ok, reason = validate_internal_authorization_header(
            request.headers.get("authorization")
        )
        if not ok:
            return JSONResponse(status_code=401, content={"detail": reason})

    user_agent = request.headers.get("user-agent", "").lower()
    if any(keyword in user_agent for keyword in SECURITY_BLOCKED_USER_AGENT_KEYWORDS):
        return JSONResponse(status_code=403, content={"detail": "Client not allowed"})

    return await call_next(request)
