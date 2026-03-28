"""Тесты для Pydantic моделей API контрактов (ARCH-01)."""

import uuid

import pytest
from pydantic import ValidationError

from router_api.models import (
    RouterRequest,
    RouterDecision,
    RouterResponse,
    RouteType,
    RagMode,
    RiskLevel,
)


# ── RouterRequest ──

class TestRouterRequest:
    def test_valid_minimal(self):
        r = RouterRequest(query="Что такое ACID?")
        assert r.query == "Что такое ACID?"
        assert r.risk_level == RiskLevel.LOW
        assert r.tenant == "default"
        uuid.UUID(r.request_id)  # не бросает

    def test_valid_full(self):
        r = RouterRequest(
            request_id=str(uuid.uuid4()),
            query="Найди расхождения в документах",
            history_summary="Предыдущий диалог о проекте",
            attachments=[{"type": "file", "name": "report.pdf"}],
            tenant="company_a",
            require_json=True,
            risk_level=RiskLevel.HIGH,
            budget_class="high",
            source_hint="internal",
        )
        assert r.risk_level == RiskLevel.HIGH
        assert r.require_json is True
        assert len(r.attachments) == 1

    def test_missing_query(self):
        with pytest.raises(ValidationError):
            RouterRequest()

    def test_empty_query(self):
        with pytest.raises(ValidationError):
            RouterRequest(query="")

    def test_invalid_uuid(self):
        with pytest.raises(ValidationError, match="valid UUID"):
            RouterRequest(request_id="not-a-uuid", query="test")

    def test_invalid_risk_level(self):
        with pytest.raises(ValidationError):
            RouterRequest(query="test", risk_level="critical")

    def test_query_max_length(self):
        long_query = "x" * 32769
        with pytest.raises(ValidationError):
            RouterRequest(query=long_query)

    def test_query_at_max_length(self):
        r = RouterRequest(query="x" * 32768)
        assert len(r.query) == 32768


# ── RouterDecision ──

class TestRouterDecision:
    def test_defaults(self):
        d = RouterDecision()
        assert d.route == RouteType.DIRECT_LOCAL
        assert d.rag_mode == RagMode.NONE
        assert d.cloud_allowed is True
        assert d.reason_codes == []

    def test_valid_decision(self):
        d = RouterDecision(
            route="local_with_rag",
            rag_mode="deep",
            cloud_allowed=False,
            reason_codes=["internal_knowledge_detected", "keyword_match"],
        )
        assert d.route == RouteType.LOCAL_WITH_RAG
        assert d.rag_mode == RagMode.DEEP

    def test_invalid_reason_code(self):
        with pytest.raises(ValidationError, match="Unknown reason codes"):
            RouterDecision(reason_codes=["made_up_code"])

    def test_all_route_types(self):
        for rt in RouteType:
            d = RouterDecision(route=rt)
            assert d.route == rt


# ── RouterResponse ──

class TestRouterResponse:
    def test_valid_response(self):
        r = RouterResponse(
            request_id=str(uuid.uuid4()),
            answer="ACID — это набор свойств транзакций.",
            confidence=0.85,
            grounded=True,
        )
        assert r.confidence == 0.85
        assert r.escalated_to_cloud is False

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            RouterResponse(request_id="x", confidence=1.5)
        with pytest.raises(ValidationError):
            RouterResponse(request_id="x", confidence=-0.1)

    def test_public_response_hides_internals(self):
        r = RouterResponse(
            request_id="test-id",
            answer="answer",
            route=RouteType.DIRECT_LOCAL,
            rag_mode=RagMode.NONE,
            reason_codes=["default_route"],
            latency_ms=150.0,
        )
        pub = r.public_response()
        assert "route" not in pub
        assert "reason_codes" not in pub
        assert "latency_ms" not in pub
        assert pub["answer"] == "answer"
        assert pub["request_id"] == "test-id"

    def test_json_schema_has_required_fields(self):
        schema = RouterResponse.model_json_schema()
        props = schema["properties"]
        assert "request_id" in props
        assert "answer" in props
        assert "confidence" in props
        assert "grounded" in props
        assert "used_context" in props
        assert "repair_applied" in props
        assert "escalated_to_cloud" in props
