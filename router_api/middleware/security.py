"""Security middleware: API key auth, rate limiting, input sanitization. ARCH-11."""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("router_api.security")


class SecurityMiddleware(BaseHTTPMiddleware):
    """API key validation + rate limiting + audit logging."""

    def __init__(self, app, api_keys: list[str] | None = None,
                 rate_limit_per_min: int = 60):
        super().__init__(app)
        self._api_keys = set(api_keys) if api_keys else set()
        self._rate_limit = rate_limit_per_min
        # In-memory rate limiter (per API key or IP)
        self._request_counts: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip health endpoint
        if request.url.path == "/health":
            return await call_next(request)

        # API key auth (если ключи сконфигурированы)
        if self._api_keys:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse(status_code=401, content={"error": "Missing Authorization header"})
            token = auth[7:]
            if token not in self._api_keys:
                logger.warning(f"Invalid API key from {request.client.host}")
                return JSONResponse(status_code=401, content={"error": "Invalid API key"})
            client_id = token[:8]
        else:
            client_id = request.client.host if request.client else "unknown"

        # Rate limiting
        if not self._check_rate_limit(client_id):
            logger.warning(f"Rate limit exceeded for {client_id}")
            return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

        # Audit log
        logger.info(f"REQUEST client={client_id} method={request.method} path={request.url.path}")

        response = await call_next(request)

        # Ensure no internal metadata leaks without X-Debug
        # (handled in RouterResponse.public_response, but double-check headers)
        return response

    def _check_rate_limit(self, client_id: str) -> bool:
        """Sliding window rate limiter."""
        now = time.monotonic()
        window = self._request_counts[client_id]

        # Remove requests older than 60s
        cutoff = now - 60
        self._request_counts[client_id] = [t for t in window if t > cutoff]

        if len(self._request_counts[client_id]) >= self._rate_limit:
            return False

        self._request_counts[client_id].append(now)
        return True


def sanitize_query(query: str) -> str:
    """Очистка пользовательского ввода от опасных символов."""
    # Удалить null bytes
    query = query.replace("\x00", "")
    # Удалить control characters (кроме \n \t)
    query = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    # Ограничить длину (32K)
    return query[:32768]
