import os
from dotenv import load_dotenv

load_dotenv()


# Token expiry settings (loaded from .env)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRY_MINUTES", 20))
REFRESH_TOKEN_EXPIRE_HOURS = int(os.getenv("REFRESH_TOKEN_EXPIRY_HOURS", 24))
SESSION_INACTIVITY_TIMEOUT_MINUTES = int(
    os.getenv("SESSION_INACTIVITY_TIMEOUT_MINUTES", 30)
)

# Cookie settings for refresh token (backend only)
REFRESH_TOKEN_COOKIE_NAME = "refresh_token"
REFRESH_TOKEN_COOKIE_SECURE = (
    os.getenv("REFRESH_TOKEN_COOKIE_SECURE", "false").lower() == "true"
)  # Set to true in production with HTTPS
REFRESH_TOKEN_COOKIE_HTTPONLY = True  # Always true - prevents JS access
REFRESH_TOKEN_COOKIE_SAMESITE = os.getenv(
    "REFRESH_TOKEN_COOKIE_SAMESITE", "lax"
)  # 'lax', 'strict', or 'none'

# CORS settings - specify allowed frontend origins (comma-separated in .env)
# Example: CORS_ORIGINS=http://localhost:5173,http://192.168.3.235:5173
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://192.168.3.235:5173").split(",")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY")
URL_DATABASE = os.getenv("URL_DATABASE")
BASE_DIR = os.getenv("BASE_DIR")
BASE_URL_SUBCLASSIFICATION = os.getenv("BASE_URL_SUBCLASSIFICATION")
BASE_URL_DETECTION = os.getenv("BASE_URL_DETECTION")

# Internal Service API Key for service-to-service authentication
# This key must be the same across all internal services (backend, yolo12, subclassification)
# Generate a secure key using: python -c "import secrets; print(secrets.token_urlsafe(64))"
INTERNAL_SERVICE_API_KEY = os.getenv(
    "INTERNAL_SERVICE_API_KEY", "default-internal-service-key-change-in-production"
)
INTERNAL_SERVICE_API_KEY_HEADER = "X-Internal-Service-Key"
