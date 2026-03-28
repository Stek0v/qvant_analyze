"""Error handling middleware. ARCH-10.

Ловит все исключения и возвращает валидный RouterResponse.
Никогда не возвращает raw 500 — всегда degraded response.
"""

from __future__ import annotations

import logging
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from router_api.exceptions import RouterError

logger = logging.getLogger("router_api")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except RouterError as e:
            logger.warning(f"RouterError [{e.status_code}]: {e.detail}")
            return _error_response(
                request_id=_extract_request_id(request),
                status_code=e.status_code,
                detail=e.detail,
            )
        except Exception as e:
            logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")
            return _error_response(
                request_id=_extract_request_id(request),
                status_code=500,
                detail="Internal server error",
            )


def _error_response(request_id: str, status_code: int, detail: str) -> JSONResponse:
    """Всегда возвращает структуру совместимую с RouterResponse."""
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "answer": f"Ошибка: {detail}",
            "citations": [],
            "confidence": 0.0,
            "grounded": False,
            "used_context": False,
            "repair_applied": False,
            "escalated_to_cloud": False,
        },
    )


def _extract_request_id(request: Request) -> str:
    """Попытаться извлечь request_id из тела запроса."""
    # request body уже может быть consumed — fallback на header или пустую строку
    return request.headers.get("X-Request-ID", "unknown")
