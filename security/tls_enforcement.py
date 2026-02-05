"""
HIPAA 164.312(e)(1) — Transmission Security Middleware

This module provides TLS/HTTPS enforcement for FastAPI applications to ensure
all data in transit is encrypted, complying with HIPAA transmission security
requirements.

Capabilities:
- Enforce HTTPS for all requests (reject HTTP in production)
- Detect protocol via X-Forwarded-Proto header (for reverse proxy setups)
- Add security headers (HSTS, X-Content-Type-Options, etc.)
- Provide audit logging for transmission security events

Environment Variables:
- ENFORCE_HTTPS: Set to "true" to enforce HTTPS (default: "false" for dev)
- ALLOW_LOCALHOST_HTTP: Allow HTTP for localhost in dev (default: "true")
- HSTS_MAX_AGE: HSTS max-age in seconds (default: 31536000 = 1 year)
- TRUSTED_PROXIES: Comma-separated list of trusted proxy IPs (optional)

Author: Security Engineering
HIPAA Control: 164.312(e)(1) - Transmission Security
"""

import os
import logging
from typing import Optional, Set, Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging for TLS enforcement events
logger = logging.getLogger("pebsi_security.tls")


class TLSEnforcementConfig:
    """
    Configuration for TLS enforcement.
    
    Loaded from environment variables with secure defaults.
    """
    
    def __init__(self):
        # Whether to enforce HTTPS (should be "true" in production)
        self.enforce_https: bool = os.getenv("ENFORCE_HTTPS", "false").lower() == "true"
        
        # Allow HTTP for localhost/development environments
        self.allow_localhost_http: bool = os.getenv("ALLOW_LOCALHOST_HTTP", "true").lower() == "true"
        
        # HSTS max-age header (1 year default)
        self.hsts_max_age: int = int(os.getenv("HSTS_MAX_AGE", "31536000"))
        
        # Include subdomains in HSTS
        self.hsts_include_subdomains: bool = os.getenv("HSTS_INCLUDE_SUBDOMAINS", "true").lower() == "true"
        
        # HSTS preload (only enable if domain is on HSTS preload list)
        self.hsts_preload: bool = os.getenv("HSTS_PRELOAD", "false").lower() == "true"
        
        # Trusted proxy IPs that can set X-Forwarded-Proto
        trusted_proxies_raw = os.getenv("TRUSTED_PROXIES", "")
        self.trusted_proxies: Set[str] = set(
            p.strip() for p in trusted_proxies_raw.split(",") if p.strip()
        ) if trusted_proxies_raw else set()
        
        # Localhost identifiers for development bypass
        self.localhost_hosts: Set[str] = {
            "localhost",
            "127.0.0.1",
            "::1",
            "0.0.0.0"
        }
        
        # Paths that are excluded from HTTPS enforcement (e.g., health checks)
        excluded_paths_raw = os.getenv("TLS_EXCLUDED_PATHS", "/health,/ready,/live")
        self.excluded_paths: Set[str] = set(
            p.strip() for p in excluded_paths_raw.split(",") if p.strip()
        )


class TLSEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce TLS/HTTPS for all requests.
    
    HIPAA 164.312(e)(1) Compliance:
    - Rejects non-HTTPS requests in production
    - Detects protocol via X-Forwarded-Proto header for reverse proxy setups
    - Adds security headers (HSTS, X-Content-Type-Options, etc.)
    - Logs transmission security events for audit trail
    
    Usage:
        from security.tls_enforcement import TLSEnforcementMiddleware
        app.add_middleware(TLSEnforcementMiddleware)
    """
    
    def __init__(self, app, config: Optional[TLSEnforcementConfig] = None):
        super().__init__(app)
        self.config = config or TLSEnforcementConfig()
        
        if self.config.enforce_https:
            logger.info("TLS enforcement ENABLED - HTTP requests will be rejected in production")
        else:
            logger.warning("TLS enforcement DISABLED - Enable for production with ENFORCE_HTTPS=true")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, checking X-Forwarded-For header."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _get_protocol(self, request: Request) -> str:
        """Determine the actual protocol of the request."""
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "").lower()
        
        if forwarded_proto in ("http", "https"):
            client_ip = self._get_client_ip(request)
            if self.config.trusted_proxies:
                if client_ip not in self.config.trusted_proxies:
                    logger.warning(f"X-Forwarded-Proto from untrusted source: {client_ip}")
                    return str(request.url.scheme)
            return forwarded_proto
        
        return str(request.url.scheme)
    
    def _is_localhost(self, request: Request) -> bool:
        """Check if the request is from localhost."""
        host = request.headers.get("Host", "").split(":")[0].lower()
        client_ip = self._get_client_ip(request)
        
        return (
            host in self.config.localhost_hosts or
            client_ip in self.config.localhost_hosts
        )
    
    def _is_excluded_path(self, request: Request) -> bool:
        """Check if the request path is excluded from TLS enforcement."""
        path = request.url.path.lower()
        return any(path.startswith(excluded) for excluded in self.config.excluded_paths)
    
    def _build_hsts_header(self) -> str:
        """Build HSTS header value based on configuration."""
        hsts_value = f"max-age={self.config.hsts_max_age}"
        if self.config.hsts_include_subdomains:
            hsts_value += "; includeSubDomains"
        if self.config.hsts_preload:
            hsts_value += "; preload"
        return hsts_value
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to the response."""
        if self.config.enforce_https:
            response.headers["Strict-Transport-Security"] = self._build_hsts_header()
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and enforce TLS requirements."""
        protocol = self._get_protocol(request)
        client_ip = self._get_client_ip(request)
        
        if self.config.enforce_https:
            is_secure = protocol == "https"
            is_localhost = self._is_localhost(request)
            is_excluded = self._is_excluded_path(request)
            
            should_block = (
                not is_secure and
                not is_excluded and
                not (self.config.allow_localhost_http and is_localhost)
            )
            
            if should_block:
                logger.warning(
                    f"BLOCKED insecure request: {client_ip} - {request.method} {request.url.path} "
                    f"- Protocol: {protocol}"
                )
                
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "HTTPS Required",
                        "detail": "This endpoint requires a secure HTTPS connection.",
                        "hipaa_control": "164.312(e)(1) - Transmission Security"
                    }
                )
        
        response = await call_next(request)
        self._add_security_headers(response)
        
        return response


def validate_tls_configuration() -> dict:
    """Validate TLS configuration and return a status report."""
    config = TLSEnforcementConfig()
    
    issues = []
    warnings = []
    
    if not config.enforce_https:
        warnings.append("ENFORCE_HTTPS is not enabled - Enable for production")
    
    if config.allow_localhost_http and config.enforce_https:
        warnings.append("ALLOW_LOCALHOST_HTTP is enabled - Disable for strict production")
    
    if config.hsts_max_age < 31536000:
        warnings.append(f"HSTS_MAX_AGE ({config.hsts_max_age}) is less than recommended 1 year")
    
    return {
        "enforce_https": config.enforce_https,
        "allow_localhost_http": config.allow_localhost_http,
        "hsts_max_age": config.hsts_max_age,
        "issues": issues,
        "warnings": warnings,
        "compliant": len(issues) == 0 and config.enforce_https
    }
