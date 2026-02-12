import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

from config import (
    INTERNAL_SERVICE_API_KEY,
    INTERNAL_SERVICE_API_KEY_HEADER,
    SECURITY_BLOCKED_USER_AGENT_KEYWORDS,
)

EXEMPT_PATHS = {"/docs", "/redoc", "/openapi.json"}


def _has_valid_internal_key(request: Request) -> bool:
    api_key = request.headers.get(INTERNAL_SERVICE_API_KEY_HEADER)
    if not api_key:
        return False
    return secrets.compare_digest(api_key, INTERNAL_SERVICE_API_KEY)


async def security_middleware(request: Request, call_next):
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    user_agent = request.headers.get("user-agent", "").lower()
    if any(keyword in user_agent for keyword in SECURITY_BLOCKED_USER_AGENT_KEYWORDS):
        return JSONResponse(status_code=403, content={"detail": "Client not allowed"})

    if not _has_valid_internal_key(request):
        return JSONResponse(
            status_code=403,
            content={"detail": "Client not allowed"},
        )

    return await call_next(request)
