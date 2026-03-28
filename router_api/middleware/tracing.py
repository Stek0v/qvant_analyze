"""Langfuse tracing middleware. Trace каждый запрос через Router API.

Записывает: route, rag_mode, latency, tokens, cost, reason_codes.
Langfuse UI: http://10.23.0.64:3000
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("router_api.tracing")

# Lazy init — Langfuse может быть недоступен
_langfuse = None


def _get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            host=os.getenv("ROUTER_LANGFUSE_HOST", "http://127.0.0.1:3000"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-qvant"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-qvant"),
        )
        logger.info("Langfuse tracing initialized")
    except Exception as e:
        logger.warning(f"Langfuse unavailable: {e}")
        _langfuse = False  # Don't retry
    return _langfuse


def trace_request(
    request_id: str,
    query: str,
    route: str,
    rag_mode: str,
    reason_codes: list[str],
    latency_ms: float,
    tokens_generated: int = 0,
    cost_usd: float = 0.0,
    escalated: bool = False,
    repair: bool = False,
    confidence: float = 0.0,
    answer_preview: str = "",
):
    """Отправить trace в Langfuse (async-safe, fire-and-forget)."""
    lf = _get_langfuse()
    if not lf:
        return

    try:
        trace = lf.trace(
            id=request_id,
            name="router_request",
            input={"query": query[:500]},
            output={"answer_preview": answer_preview[:200]},
            metadata={
                "route": route,
                "rag_mode": rag_mode,
                "reason_codes": reason_codes,
                "escalated_to_cloud": escalated,
                "repair_applied": repair,
                "confidence": confidence,
            },
            tags=[route, rag_mode] + (["cloud"] if escalated else []) + (["repair"] if repair else []),
        )

        # Generation span
        trace.generation(
            name="llm_generation",
            model="qwen3.5-27b-q4_k_m",
            usage={
                "total_tokens": tokens_generated,
            },
            metadata={
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            },
        )

        lf.flush()
    except Exception as e:
        logger.debug(f"Langfuse trace failed: {e}")
