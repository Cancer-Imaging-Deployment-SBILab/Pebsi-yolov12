import os
from dotenv import load_dotenv

load_dotenv()

SECURITY_BLOCKED_USER_AGENT_KEYWORDS = {
    item.strip().lower()
    for item in os.getenv(
        "SECURITY_BLOCKED_USER_AGENTS",
        "curl,postman,wget,httpx,insomnia,libwww,java",
    ).split(",")
    if item.strip()
}

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY")
URL_DATABASE = os.getenv("URL_DATABASE")
BASE_DIR = os.getenv("BASE_DIR")


# Internal Service API Key for service-to-service authentication
# This key must be the same across all internal services (backend, yolo12, subclassification)
# Generate a secure key using: python -c "import secrets; print(secrets.token_urlsafe(64))"
INTERNAL_SERVICE_API_KEY = os.getenv(
    "INTERNAL_SERVICE_API_KEY", "default-internal-service-key-change-in-production"
)
INTERNAL_SERVICE_API_KEY_HEADER = "X-Internal-Service-Key"
