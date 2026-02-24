import os
from dotenv import load_dotenv

load_dotenv()

SECURITY_BLOCKED_USER_AGENT_KEYWORDS = {
    item.strip().lower()
    for item in os.getenv(
        "SECURITY_BLOCKED_USER_AGENTS",
        "postman,wget,httpx,insomnia,libwww,java",
    ).split(",")
    if item.strip()
}

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY")
URL_DATABASE = os.getenv("URL_DATABASE")
BASE_DIR = os.getenv("BASE_DIR")

# -----------------------------------------------------------------------------
# Internal service-to-service JWT (optional, in addition to mTLS)
# -----------------------------------------------------------------------------
INTERNAL_JWT_ENABLED = True
INTERNAL_JWT_SECRET = os.getenv("INTERNAL_JWT_SECRET", "")
INTERNAL_JWT_ISSUER = os.getenv("INTERNAL_JWT_ISSUER", "pebsi-backend")
INTERNAL_JWT_AUDIENCE = os.getenv("INTERNAL_JWT_AUDIENCE", "pebsi-internal")
INTERNAL_JWT_ALGORITHM = os.getenv("INTERNAL_JWT_ALGORITHM", "HS256")

SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8001"))
SERVICE_REFRESH = os.getenv("SERVICE_REFRESH", "false").lower() == "true"
MTLS_ENABLED = True
SSL_REQUIRE_CLIENT_CERT = True
SSL_CA_FILE = os.getenv("SSL_CA_FILE", "")
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", "")
SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", "")


def _require_nonempty(value: str, name: str) -> None:
    if not (value or "").strip():
        raise RuntimeError(f"{name} must be set")


if INTERNAL_JWT_ENABLED:
    _require_nonempty(INTERNAL_JWT_SECRET, "INTERNAL_JWT_SECRET")

if MTLS_ENABLED:
    _require_nonempty(SSL_CA_FILE, "SSL_CA_FILE")
    _require_nonempty(SSL_CERT_FILE, "SSL_CERT_FILE")
    _require_nonempty(SSL_KEY_FILE, "SSL_KEY_FILE")
