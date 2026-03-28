"""Edge case tests: tie-breaks, boundary thresholds, empty/malformed inputs.

Запуск: pytest router_api/tests/test_edge_cases.py -v
"""

from __future__ import annotations

import pytest

from router_api.config import RouterConfig
from router_api.models import (
    RouterRequest,
    RouteType,
    RagMode,
    RetrievalResult,
    LlamaResponse,
)
from router_api.routers.pipeline_router import PipelineRouter
from router_api.routers.rag_router import RAGRouter
from router_api.routers.escalation_router import EscalationRouter
from router_api.middleware.security import sanitize_query


@pytest.fixture
def config():
    return RouterConfig()


# ═══════════════════════════════════════════════════════════
# RAG Router boundary thresholds
# ═══════════════════════════════════════════════════════════

class TestRAGBoundaries:
    def test_score_just_below_shallow_threshold(self, config):
        """0.779 < 0.78 → deep, not shallow."""
        router = RAGRouter(config)
        results = [RetrievalResult(content="doc", score=0.779), RetrievalResult(content="doc2", score=0.50)]
        d = router.decide(results)
        assert d.mode == RagMode.DEEP  # below 0.78

    def test_score_just_above_deep_threshold(self, config):
        """0.601 >= 0.60 → deep."""
        router = RAGRouter(config)
        results = [RetrievalResult(content="doc", score=0.601), RetrievalResult(content="doc2", score=0.50)]
        d = router.decide(results)
        assert d.mode == RagMode.DEEP

    def test_score_just_below_deep_threshold(self, config):
        """0.599 < 0.60 → none."""
        router = RAGRouter(config)
        results = [RetrievalResult(content="doc", score=0.599), RetrievalResult(content="doc2", score=0.50)]
        d = router.decide(results)
        assert d.mode == RagMode.NONE

    def test_tie_top1_equals_top2(self, config):
        """top1 == top2 → gap=0 → deep (gap < 0.08)."""
        router = RAGRouter(config)
        results = [RetrievalResult(content="doc1", score=0.85), RetrievalResult(content="doc2", score=0.85)]
        d = router.decide(results)
        # Score >= 0.78 BUT gap = 0 < 0.08 → falls to deep range check
        assert d.mode == RagMode.DEEP

    def test_all_scores_identical_below_threshold(self, config):
        """Все scores одинаковые и ниже порога → none."""
        router = RAGRouter(config)
        results = [RetrievalResult(content=f"doc{i}", score=0.30) for i in range(5)]
        d = router.decide(results)
        assert d.mode == RagMode.NONE

    def test_single_perfect_score(self, config):
        """Один результат с score=1.0."""
        router = RAGRouter(config)
        results = [RetrievalResult(content="doc", score=1.0)]
        d = router.decide(results)
        assert d.mode == RagMode.SHALLOW

    def test_zero_score_results(self, config):
        """Все scores = 0."""
        router = RAGRouter(config)
        results = [RetrievalResult(content="doc", score=0.0)]
        d = router.decide(results)
        assert d.mode == RagMode.NONE


# ═══════════════════════════════════════════════════════════
# Pipeline Router edge cases
# ═══════════════════════════════════════════════════════════

class TestPipelineEdgeCases:
    def test_very_long_query(self, config):
        """32K символов — не ломается."""
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="x" * 32000))
        assert d.route == RouteType.DIRECT_LOCAL

    def test_query_with_all_keywords(self, config):
        """Запрос содержит ВСЕ RAG keywords — не дублирует reason_codes."""
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="Найди в базе по документу в проекте"))
        assert d.route == RouteType.LOCAL_WITH_RAG
        assert d.reason_codes.count("keyword_match") == 1  # не дублируется

    def test_unicode_query(self, config):
        """Запрос с эмодзи и спецсимволами."""
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="🔥 Что такое Docker? 你好"))
        assert d.route == RouteType.DIRECT_LOCAL

    def test_high_risk_overrides_json(self, config):
        """risk=high + require_json → cloud (risk > json)."""
        router = PipelineRouter(config)
        d = router.route(RouterRequest(query="test", risk_level="high", require_json=True))
        assert d.route == RouteType.CLOUD_ESCALATION


# ═══════════════════════════════════════════════════════════
# Escalation Router edge cases
# ═══════════════════════════════════════════════════════════

class TestEscalationEdgeCases:
    def test_valid_json_with_extra_whitespace(self, config):
        """JSON с пробелами и переводами строк — всё ещё valid."""
        router = EscalationRouter(config)
        req = RouterRequest(query="test", require_json=True)
        resp = LlamaResponse(content='  \n  { "key": "value" }  \n  ', confidence=0.8)
        assert router.decide(req, resp) == "ok"

    def test_json_array_is_valid(self, config):
        """JSON array длиннее 10 chars — valid JSON, ok."""
        router = EscalationRouter(config)
        req = RouterRequest(query="test", require_json=True)
        resp = LlamaResponse(content='[{"id": 1}, {"id": 2}, {"id": 3}]', confidence=0.8)
        assert router.decide(req, resp) == "ok"

    def test_markdown_wrapped_json_is_invalid(self, config):
        """JSON обёрнутый в markdown → invalid → repair."""
        router = EscalationRouter(config)
        req = RouterRequest(query="test", require_json=True)
        resp = LlamaResponse(content='```json\n{"key": "value"}\n```', confidence=0.8)
        assert router.decide(req, resp, repair_count=0) == "repair"

    def test_confidence_zero_triggers_repair(self, config):
        """confidence=0.0 → repair."""
        router = EscalationRouter(config)
        req = RouterRequest(query="test")
        resp = LlamaResponse(content="Some answer here enough length", confidence=0.0)
        assert router.decide(req, resp, repair_count=0) == "repair"


# ═══════════════════════════════════════════════════════════
# Input sanitization
# ═══════════════════════════════════════════════════════════

class TestSanitization:
    def test_null_bytes_removed(self):
        assert "\x00" not in sanitize_query("test\x00query")

    def test_control_chars_removed(self):
        result = sanitize_query("test\x01\x02\x03query")
        assert result == "testquery"

    def test_newlines_preserved(self):
        result = sanitize_query("line1\nline2\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_length_truncated(self):
        result = sanitize_query("x" * 40000)
        assert len(result) == 32768
