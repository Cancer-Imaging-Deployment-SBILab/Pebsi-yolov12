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

SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8001"))
SERVICE_REFRESH = os.getenv("SERVICE_REFRESH", "false").lower() == "true"
MTLS_ENABLED = os.getenv("MTLS_ENABLED", "true").lower() == "true"
SSL_REQUIRE_CLIENT_CERT = os.getenv("SSL_REQUIRE_CLIENT_CERT", "true").lower() == "true"
SSL_CA_FILE = os.getenv("SSL_CA_FILE", "")
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", "")
SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", "")
