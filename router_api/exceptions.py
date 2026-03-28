"""Exception hierarchy для Router API. ARCH-10."""

from __future__ import annotations


class RouterError(Exception):
    """Базовое исключение Router API."""
    status_code: int = 500
    detail: str = "Internal router error"

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class BackendTimeoutError(RouterError):
    """Таймаут бэкенда (llama-server, Cognee)."""
    status_code = 502
    detail = "Backend timeout"


class BackendUnavailableError(RouterError):
    """Бэкенд недоступен."""
    status_code = 503
    detail = "Backend unavailable"


class RetrievalError(RouterError):
    """Ошибка RAG retrieval."""
    status_code = 502
    detail = "Retrieval failed"


class CloudBudgetExceededError(RouterError):
    """Превышен бюджет cloud escalation."""
    status_code = 429
    detail = "Cloud budget exceeded"


class CloudCircuitOpenError(RouterError):
    """Circuit breaker cloud открыт."""
    status_code = 503
    detail = "Cloud circuit breaker open"


class SchemaValidationError(RouterError):
    """Ответ не прошёл JSON schema validation."""
    status_code = 502
    detail = "Schema validation failed"


class RateLimitError(RouterError):
    """Превышен rate limit."""
    status_code = 429
    detail = "Rate limit exceeded"
