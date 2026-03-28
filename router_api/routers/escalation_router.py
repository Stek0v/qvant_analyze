"""Router 3 — Escalation Router (rule-based). ARCH-05.

Решает: ok | repair | cloud.
"""

from __future__ import annotations

import json

from router_api.config import RouterConfig
from router_api.models import (
    EscalationDecision,
    LlamaResponse,
    RetrievalResult,
    RouterRequest,
    RiskLevel,
)


class EscalationRouter:
    def __init__(self, config: RouterConfig):
        self._enable_cloud = config.enable_cloud

    def decide(
        self,
        request: RouterRequest,
        response: LlamaResponse,
        retrieval_results: list[RetrievalResult] | None = None,
        repair_count: int = 0,
    ) -> EscalationDecision:
        # Hard rule: high risk → cloud
        if request.risk_level == RiskLevel.HIGH and self._enable_cloud:
            return EscalationDecision.CLOUD

        # JSON validation failure
        if request.require_json and not self._is_valid_json(response.content):
            if repair_count < 1:
                return EscalationDecision.REPAIR
            return EscalationDecision.CLOUD if self._enable_cloud else EscalationDecision.OK

        # Empty or too short response
        if len(response.content.strip()) < 10:
            if repair_count < 1:
                return EscalationDecision.REPAIR
            return EscalationDecision.CLOUD if self._enable_cloud else EscalationDecision.OK

        # Low confidence
        if response.confidence < 0.3:
            if repair_count < 1:
                return EscalationDecision.REPAIR
            return EscalationDecision.CLOUD if self._enable_cloud else EscalationDecision.OK

        # Context conflict: response doesn't reference retrieved docs
        if retrieval_results and not response.content.strip():
            if repair_count < 1:
                return EscalationDecision.REPAIR
            return EscalationDecision.CLOUD if self._enable_cloud else EscalationDecision.OK

        return EscalationDecision.OK

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
