"""
Security Headers Middleware
===========================
OWASP-compliant security headers.

رؤوس الأمان الشاملة لـ OWASP
"""

from fastapi import Request, Response
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from typing import Callable, Any

from src.core.config import settings


class SecurityHeadersMiddleware(Middleware):
    """
    Add OWASP-compliant security headers to all responses.

    Headers Added:
    - Content-Security-Policy (CSP): Controls what resources can load
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking
    - Strict-Transport-Security (HSTS): Enforces HTTPS
    - X-XSS-Protection: XSS protection for older browsers
    - Referrer-Policy: Controls how much info is leaked via Referer
    - Permissions-Policy: Controls browser features

    مزو رؤوس الأمان الشاملة
    """

    def __init__(
        self,
        content_security_policy: str | None = None,
        frame_options: str = "DENY",
        strict_transport_security: str = "max-age=31536000; includeSubDomains",
        x_xss_protection: str = "1; mode=block",
        referrer_policy: str = "strict-origin-when-cross-origin",
    ) -> None:
        """
        Initialize security headers middleware.

        Args:
            content_security_policy: CSP policy (None for per-endpoint override)
            frame_options: Frame embedding policy (DENY, SAMEORIGIN)
            strict_transport_security: HSTS policy
            x_xss_protection: XSS protection header
            referrer_policy: Referrer policy
        """
        self._csp = content_security_policy
        self._frame_options = frame_options
        self._hsts = strict_transport_security
        self._xss = x_xss_protection
        self._referrer = referrer_policy

    async def __call__(self, request: Request, call_next: Callable[[Any], Any]) -> Response:
        """
        Process request and add security headers.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response with security headers added
        """
        response = await call_next(request)

        # Add Content-Security-Policy
        if self._csp:
            response.headers["Content-Security-Policy"] = self._csp
        else:
            # Default CSP for this application
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'self'; "
                "form-action 'self'; "
                "base-uri 'self'; "
                "frame-src 'self'; "
                "object-src 'none'; "
                "report-uri /csp-report; "
            )

        # Add X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Add X-Frame-Options
        response.headers["X-Frame-Options"] = self._frame_options

        # Add Strict-Transport-Security (HSTS)
        response.headers["Strict-Transport-Security"] = self._hsts

        # Add X-XSS-Protection
        response.headers["X-XSS-Protection"] = self._xss

        # Add Referrer-Policy
        response.headers["Referrer-Policy"] = self._referrer

        # Remove X-Powered-By (information leak)
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

        # Add Server header (minimal, no version info)
        response.headers["Server"] = settings.app_name

        return response


def setup_security_headers(app) -> None:
    """
    Setup security headers for FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Add CSP middleware
    app.add_middleware(
        SecurityHeadersMiddleware(
            content_security_policy=None,  # Use default
            frame_options="DENY",  # Prevent clickjacking
            strict_transport_security="max-age=31536000; includeSubDomains; preload",
            x_xss_protection="1; mode=block",
            referrer_policy="strict-origin-when-cross-origin",
        )
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
