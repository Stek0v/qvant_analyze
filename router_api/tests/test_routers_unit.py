"""Unit тесты для классов роутеров БЕЗ HTTP (CI-friendly).

Запуск: pytest router_api/tests/test_routers_unit.py -v
Не требует запущенного Router API или llama-server.
"""

from __future__ import annotations

import pytest

from router_api.config import RouterConfig
from router_api.models import (
    RouterRequest,
    RouteType,
    RagMode,
    RiskLevel,
    SourceHint,
    RetrievalResult,
    LlamaResponse,
)
from router_api.routers.pipeline_router import PipelineRouter
from router_api.routers.rag_router import RAGRouter, RagDecision
from router_api.routers.escalation_router import EscalationRouter


@pytest.fixture
def config():
    return RouterConfig()


# ═══════════════════════════════════════════════════════════
# PipelineRouter
# ═══════════════════════════════════════════════════════════

class TestPipelineRouter:
    def test_default_route_is_direct_local(self, config):
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="Hello world"))
        assert d.route == RouteType.DIRECT_LOCAL
        assert "default_route" in d.reason_codes

    def test_high_risk_goes_cloud(self, config):
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="Anything", risk_level=RiskLevel.HIGH))
        assert d.route == RouteType.CLOUD_ESCALATION
        assert "high_risk" in d.reason_codes

    def test_source_hint_internal_triggers_rag(self, config):
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="Show data", source_hint=SourceHint.INTERNAL))
        assert d.route == RouteType.LOCAL_WITH_RAG
        assert "source_hint_internal" in d.reason_codes

    def test_source_hint_mixed_triggers_rag(self, config):
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="Mixed query", source_hint=SourceHint.MIXED))
        assert d.route == RouteType.LOCAL_WITH_RAG

    def test_keyword_match_triggers_rag(self, config):
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="Найди в базе информацию"))
        assert d.route == RouteType.LOCAL_WITH_RAG
        assert "keyword_match" in d.reason_codes

    def test_deep_rag_keywords(self, config):
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="Сравни версии документации", source_hint=SourceHint.INTERNAL))
        assert d.rag_mode == RagMode.DEEP

    def test_require_json_goes_repair(self, config):
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="Extract data", require_json=True))
        assert d.route == RouteType.LOCAL_WITH_RAG_AND_REPAIR
        assert "format_strict" in d.reason_codes

    def test_rag_disabled_never_triggers(self, config):
        cfg = config.model_copy(update={"enable_rag": False})
        router = PipelineRouter(cfg)
        d = router.route(RouterRequest(query="Найди в нашей базе", source_hint=SourceHint.INTERNAL))
        assert d.route == RouteType.DIRECT_LOCAL

    def test_no_keyword_match_general(self, config):
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="What is the capital of France?"))
        assert d.route == RouteType.DIRECT_LOCAL


# ═══════════════════════════════════════════════════════════
# RAGRouter
# ═══════════════════════════════════════════════════════════

class TestRAGRouter:
    def test_empty_results_returns_none(self, config):
        router = RAGRouter(config)
        d = router.decide([])
        assert d.mode == RagMode.NONE

    def test_high_score_high_gap_shallow(self, config):
        router = RAGRouter(config)
        results = [
            RetrievalResult(content="doc1", score=0.85),
            RetrievalResult(content="doc2", score=0.60),
        ]
        d = router.decide(results)
        assert d.mode == RagMode.SHALLOW

    def test_medium_score_deep(self, config):
        router = RAGRouter(config)
        results = [
            RetrievalResult(content="doc1", score=0.70),
            RetrievalResult(content="doc2", score=0.65),
        ]
        d = router.decide(results)
        assert d.mode == RagMode.DEEP

    def test_low_score_returns_none(self, config):
        router = RAGRouter(config)
        results = [
            RetrievalResult(content="doc1", score=0.30),
            RetrievalResult(content="doc2", score=0.20),
        ]
        d = router.decide(results)
        assert d.mode == RagMode.NONE

    def test_single_result_uses_score_as_gap(self, config):
        router = RAGRouter(config)
        results = [RetrievalResult(content="doc1", score=0.90)]
        d = router.decide(results)
        assert d.mode == RagMode.SHALLOW  # 0.90 >= 0.78 and gap=0.90 >= 0.08

    def test_boundary_score_exactly_078(self, config):
        """Граничное значение: score = 0.78 ровно."""
        router = RAGRouter(config)
        results = [
            RetrievalResult(content="doc1", score=0.78),
            RetrievalResult(content="doc2", score=0.50),
        ]
        d = router.decide(results)
        assert d.mode == RagMode.SHALLOW  # >= 0.78

    def test_boundary_gap_exactly_008(self, config):
        """Граничное значение: gap = 0.08 ровно."""
        router = RAGRouter(config)
        results = [
            RetrievalResult(content="doc1", score=0.80),
            RetrievalResult(content="doc2", score=0.72),  # gap = 0.08
        ]
        d = router.decide(results)
        assert d.mode == RagMode.SHALLOW


# ═══════════════════════════════════════════════════════════
# EscalationRouter
# ═══════════════════════════════════════════════════════════

class TestEscalationRouter:
    def test_ok_on_good_response(self, config):
        router = EscalationRouter(config)
        req = RouterRequest(query="test")
        resp = LlamaResponse(content="A good detailed answer here.", confidence=0.8)
        assert router.decide(req, resp) == "ok"

    def test_repair_on_empty_response(self, config):
        router = EscalationRouter(config)
        req = RouterRequest(query="test")
        resp = LlamaResponse(content="", confidence=0.5)
        assert router.decide(req, resp, repair_count=0) == "repair"

    def test_cloud_after_repair_fails(self, config):
        router = EscalationRouter(config)
        req = RouterRequest(query="test")
        resp = LlamaResponse(content="short", confidence=0.1)
        assert router.decide(req, resp, repair_count=1) == "cloud"

    def test_repair_on_invalid_json(self, config):
        router = EscalationRouter(config)
        req = RouterRequest(query="test", require_json=True)
        resp = LlamaResponse(content="not json at all", confidence=0.7)
        assert router.decide(req, resp, repair_count=0) == "repair"

    def test_ok_on_valid_json(self, config):
        router = EscalationRouter(config)
        req = RouterRequest(query="test", require_json=True)
        resp = LlamaResponse(content='{"key": "value"}', confidence=0.8)
        assert router.decide(req, resp) == "ok"

    def test_high_risk_goes_cloud(self, config):
        router = EscalationRouter(config)
        req = RouterRequest(query="test", risk_level=RiskLevel.HIGH)
        resp = LlamaResponse(content="good answer", confidence=0.9)
        assert router.decide(req, resp) == "cloud"

    def test_cloud_disabled_returns_ok(self, config):
        cfg = config.model_copy(update={"enable_cloud": False})
        router = EscalationRouter(cfg)
        req = RouterRequest(query="test")
        resp = LlamaResponse(content="short", confidence=0.1)
        assert router.decide(req, resp, repair_count=1) == "ok"  # cloud disabled → ok

    def test_low_confidence_triggers_repair(self, config):
        router = EscalationRouter(config)
        req = RouterRequest(query="test")
        resp = LlamaResponse(content="Some answer here enough chars", confidence=0.2)
        assert router.decide(req, resp, repair_count=0) == "repair"
