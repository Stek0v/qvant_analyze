"""Router 1 — Pipeline Router (rules-based). ARCH-03.

Решает: direct_local | local_with_rag | local_with_rag_and_repair | cloud_candidate
"""

from __future__ import annotations

from router_api.config import RouterConfig
from router_api.models import (
    RouterRequest,
    RouterDecision,
    RouteType,
    RagMode,
    RiskLevel,
    SourceHint,
)


class PipelineRouter:
    def __init__(self, config: RouterConfig):
        self._keywords = [kw.lower() for kw in config.rag_keywords]
        self._enable_rag = config.enable_rag

    def route(self, request: RouterRequest) -> RouterDecision:
        reasons: list[str] = []

        # Hard rule: high risk → cloud
        if request.risk_level == RiskLevel.HIGH:
            reasons.append("high_risk")
            return RouterDecision(
                route=RouteType.CLOUD_ESCALATION,
                rag_mode=RagMode.NONE,
                cloud_allowed=True,
                reason_codes=reasons,
            )

        # Hard rule: require_json → structured path with repair
        if request.require_json:
            reasons.append("format_strict")
            rag_mode = RagMode.NONE
            route = RouteType.LOCAL_WITH_RAG_AND_REPAIR

            if self._needs_rag(request, reasons):
                rag_mode = RagMode.SHALLOW
            return RouterDecision(
                route=route, rag_mode=rag_mode,
                cloud_allowed=True, reason_codes=reasons,
            )

        # Check if RAG needed
        if self._needs_rag(request, reasons):
            # Decide shallow vs deep by keywords
            if self._needs_deep_rag(request):
                return RouterDecision(
                    route=RouteType.LOCAL_WITH_RAG,
                    rag_mode=RagMode.DEEP,
                    cloud_allowed=True,
                    reason_codes=reasons,
                )
            return RouterDecision(
                route=RouteType.LOCAL_WITH_RAG,
                rag_mode=RagMode.SHALLOW,
                cloud_allowed=True,
                reason_codes=reasons,
            )

        # Default: direct local
        reasons.append("default_route")
        return RouterDecision(
            route=RouteType.DIRECT_LOCAL,
            rag_mode=RagMode.NONE,
            cloud_allowed=True,
            reason_codes=reasons,
        )

    def _needs_rag(self, request: RouterRequest, reasons: list[str]) -> bool:
        if not self._enable_rag:
            return False

        # Source hint
        if request.source_hint in (SourceHint.INTERNAL, SourceHint.MIXED):
            reasons.append("source_hint_internal")
            return True

        # Keyword match
        query_lower = request.query.lower()
        for kw in self._keywords:
            if kw in query_lower:
                reasons.append("keyword_match")
                return True

        return False

    def _needs_deep_rag(self, request: RouterRequest) -> bool:
        deep_markers = ["сравни", "сопоставь", "в нескольких документах",
                        "найди противоречия", "итог по всем", "расхождения"]
        query_lower = request.query.lower()
        return any(m in query_lower for m in deep_markers)
