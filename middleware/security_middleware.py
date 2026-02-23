from fastapi import Request
from fastapi.responses import JSONResponse

from config import (
    MTLS_ENABLED,
    SECURITY_BLOCKED_USER_AGENT_KEYWORDS,
)

EXEMPT_PATHS = {"/docs", "/redoc", "/openapi.json"}


async def security_middleware(request: Request, call_next):
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    if MTLS_ENABLED:
        # mTLS-enabled deployments must use HTTPS transport.
        # When a client certificate is missing or invalid, TLS handshake fails
        # before this middleware runs, so this check only guards protocol bypasses.
        forwarded_proto = request.headers.get("x-forwarded-proto", "")
        proto = (forwarded_proto.split(",")[0].strip() or request.url.scheme).lower()
        if proto != "https":
            return JSONResponse(
                status_code=403,
                content={"detail": "mTLS required"},
            )

    user_agent = request.headers.get("user-agent", "").lower()
    if any(keyword in user_agent for keyword in SECURITY_BLOCKED_USER_AGENT_KEYWORDS):
        return JSONResponse(status_code=403, content={"detail": "Client not allowed"})

    return await call_next(request)
