"""FastAPI Router API — точка входа. ARCH-02.

Использование:
    uvicorn router_api.main:app --host 127.0.0.1 --port 8100
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from router_api.config import RouterConfig, load_config
from router_api.models import (
    RouterRequest,
    RouterResponse,
    RagMode,
    LlamaResponse,
    EscalationDecision,
)
from router_api.routers.pipeline_router import PipelineRouter
from router_api.routers.rag_router import RAGRouter
from router_api.routers.escalation_router import EscalationRouter
from router_api.adapters.llama_adapter import LlamaAdapter
from router_api.adapters.levara_adapter import LevaraAdapter
from router_api.adapters.cloud_adapter import CloudAdapter
from router_api.middleware.error_handler import ErrorHandlerMiddleware
from router_api.middleware.security import SecurityMiddleware, sanitize_query
from router_api.middleware.tracing import trace_request


# =====================================================================
# Application state
# =====================================================================

class AppState:
    config: RouterConfig
    pipeline_router: PipelineRouter
    rag_router: RAGRouter
    escalation_router: EscalationRouter
    llama: LlamaAdapter
    rag: LevaraAdapter
    cloud: CloudAdapter

    def __init__(self, config: RouterConfig):
        self.config = config
        self.pipeline_router = PipelineRouter(config)
        self.rag_router = RAGRouter(config)
        self.escalation_router = EscalationRouter(config)
        self.llama = LlamaAdapter(
            base_url=config.llama_url,
            timeout=config.llama_timeout_s,
            max_tokens=config.llama_max_tokens,
        )
        self.rag = LevaraAdapter(
            base_url=config.levara_url,
        )
        self.cloud = CloudAdapter(
            api_url=config.cloud_api_url,
            model=config.cloud_model,
            timeout=config.cloud_timeout_s,
            daily_budget_usd=config.cloud_daily_budget_usd,
            circuit_breaker_fails=config.cloud_circuit_breaker_fails,
            circuit_breaker_reset_s=config.cloud_circuit_breaker_reset_s,
        )

    async def close(self):
        await self.llama.close()
        await self.cloud.close()


state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    config = load_config()
    state = AppState(config)
    yield
    await state.close()
    state = None


# =====================================================================
# FastAPI app
# =====================================================================

app = FastAPI(
    title="qvant Router API",
    description="Local-first LLM routing with cascading routers",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(ErrorHandlerMiddleware)
# SecurityMiddleware добавляется в lifespan после загрузки конфига


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/health")
async def health():
    """Статус всех бэкендов."""
    checks = {
        "llama": "ok" if await state.llama.health() else "unreachable",
        "levara": "ok" if await state.rag.health() else "unreachable",
    }
    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "backends": checks}


@app.post("/v1/route", response_model=RouterResponse)
async def route_request(req: RouterRequest, raw_request: Request):
    """Главный endpoint: принять запрос, маршрутизация, ответ."""
    t0 = time.monotonic()
    debug = raw_request.headers.get("X-Debug") == "true"

    # Sanitize input
    req.query = sanitize_query(req.query)

    # Step 1: Pipeline Router
    decision = state.pipeline_router.route(req)

    # Step 2: Execute pipeline
    retrieval_results = []
    llama_resp = LlamaResponse()
    repair_count = 0
    escalated = False
    cost_usd = 0.0

    # RAG retrieval (if needed)
    if decision.rag_mode != RagMode.NONE and state.config.enable_rag:
        retrieval_results = await _do_retrieval(req.query, decision.rag_mode)

        # Re-check RAG quality
        rag_decision = state.rag_router.decide(retrieval_results)
        decision.rag_mode = rag_decision.mode
        if rag_decision.reason not in decision.reason_codes:
            decision.reason_codes.append(rag_decision.reason)

    # Local generation
    llama_resp = await _do_generate(req, retrieval_results)

    # Step 3: Escalation Router
    esc = state.escalation_router.decide(req, llama_resp, retrieval_results, repair_count)

    if esc == EscalationDecision.REPAIR:
        repair_count += 1
        decision.reason_codes.append("repair_triggered")
        llama_resp = await _do_generate(req, retrieval_results, repair=True)
        esc = state.escalation_router.decide(req, llama_resp, retrieval_results, repair_count)

    if esc == EscalationDecision.CLOUD and state.config.enable_cloud:
        decision.reason_codes.append("cloud_escalation")
        escalated = True
        # TODO: ARCH-08 — Cloud adapter

    latency_ms = (time.monotonic() - t0) * 1000

    response = RouterResponse(
        request_id=req.request_id,
        answer=llama_resp.content,
        confidence=llama_resp.confidence,
        grounded=bool(retrieval_results),
        used_context=decision.rag_mode != RagMode.NONE,
        repair_applied=repair_count > 0,
        escalated_to_cloud=escalated,
    )

    if debug:
        response.route = decision.route
        response.rag_mode = decision.rag_mode
        response.reason_codes = decision.reason_codes
        response.latency_ms = latency_ms
        response.tokens_generated = llama_resp.completion_tokens
        response.cost_usd = cost_usd

    # Langfuse trace (fire-and-forget)
    trace_request(
        request_id=req.request_id,
        query=req.query,
        route=decision.route.value,
        rag_mode=decision.rag_mode.value,
        reason_codes=decision.reason_codes,
        latency_ms=latency_ms,
        tokens_generated=llama_resp.completion_tokens,
        cost_usd=cost_usd,
        escalated=escalated,
        repair=repair_count > 0,
        confidence=llama_resp.confidence,
        answer_preview=llama_resp.content,
    )

    return response


# =====================================================================
# Internal helpers
# =====================================================================

async def _do_retrieval(query: str, mode: RagMode) -> list:
    """Retrieval через Levara vector search."""
    return await state.rag.search(query, mode=mode.value)


async def _do_generate(
    req: RouterRequest,
    context: list,
    repair: bool = False,
) -> LlamaResponse:
    """Генерация через LlamaAdapter."""
    ctx_texts = [r.content for r in context if hasattr(r, "content") and r.content]
    return await state.llama.generate_with_context(
        query=req.query,
        context=ctx_texts or None,
        history_summary=req.history_summary,
        require_json=req.require_json,
        repair=repair,
    )
