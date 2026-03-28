"""Pydantic модели API контрактов из v2+routes.md (строки 536-568)."""

from __future__ import annotations

import uuid as _uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =====================================================================
# Enums
# =====================================================================

class RouteType(StrEnum):
    DIRECT_LOCAL = "direct_local"
    LOCAL_WITH_RAG = "local_with_rag"
    LOCAL_WITH_RAG_AND_REPAIR = "local_with_rag_and_repair"
    CLOUD_ESCALATION = "cloud_escalation"


class RagMode(StrEnum):
    NONE = "none"
    SHALLOW = "shallow"
    DEEP = "deep"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BudgetClass(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class SourceHint(StrEnum):
    NONE = "none"
    INTERNAL = "internal"
    MIXED = "mixed"


ALLOWED_REASON_CODES = frozenset({
    "internal_knowledge_detected",
    "format_strict",
    "high_risk",
    "retrieval_low_confidence",
    "retrieval_high_confidence",
    "keyword_match",
    "source_hint_internal",
    "repair_triggered",
    "repair_failed",
    "cloud_budget_exceeded",
    "backend_timeout",
    "default_route",
})


# =====================================================================
# Input Contract — RouterRequest
# =====================================================================

class RouterRequest(BaseModel):
    """Входной контракт Router API."""

    request_id: str = Field(default_factory=lambda: str(_uuid.uuid4()))
    query: str = Field(..., min_length=1, max_length=32768)
    history_summary: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    tenant: str = "default"
    require_json: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    budget_class: BudgetClass = BudgetClass.NORMAL
    source_hint: SourceHint = SourceHint.NONE

    @field_validator("request_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            _uuid.UUID(v)
        except ValueError:
            raise ValueError(f"request_id must be a valid UUID, got: {v!r}")
        return v


# =====================================================================
# Router Decision Contract — RouterDecision
# =====================================================================

class RouterDecision(BaseModel):
    """Решение каскада роутеров."""

    route: RouteType = RouteType.DIRECT_LOCAL
    rag_mode: RagMode = RagMode.NONE
    cloud_allowed: bool = True
    reason_codes: list[str] = Field(default_factory=list)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, v: list[str]) -> list[str]:
        unknown = set(v) - ALLOWED_REASON_CODES
        if unknown:
            raise ValueError(f"Unknown reason codes: {unknown}")
        return v


# =====================================================================
# Response Contract — RouterResponse
# =====================================================================

class RouterResponse(BaseModel):
    """Ответ Router API клиенту."""

    request_id: str
    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    grounded: bool = True
    used_context: bool = False
    repair_applied: bool = False
    escalated_to_cloud: bool = False

    # Debug fields (скрываются без X-Debug header)
    route: RouteType | None = None
    rag_mode: RagMode | None = None
    reason_codes: list[str] | None = None
    latency_ms: float | None = None
    tokens_generated: int | None = None
    cost_usd: float | None = None

    def public_response(self) -> dict:
        """Ответ без internal metadata."""
        return self.model_dump(
            exclude={"route", "rag_mode", "reason_codes", "latency_ms",
                     "tokens_generated", "cost_usd"},
            exclude_none=True,
        )


# =====================================================================
# Internal models
# =====================================================================

class RetrievalResult(BaseModel):
    """Результат одного retrieval hit."""
    content: str
    score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LlamaResponse(BaseModel):
    """Ответ от llama-server."""
    content: str = ""
    thinking: str | None = None
    confidence: float = 0.0
    tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CloudResponse(BaseModel):
    """Ответ от cloud provider."""
    content: str = ""
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class VerificationResult(BaseModel):
    """Результат локальной верификации (Iter 2)."""
    grounded: bool = True
    format_ok: bool = True
    complete: bool = True
    gaps: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class EscalationDecision(StrEnum):
    OK = "ok"
    REPAIR = "repair"
    CLOUD = "cloud"
