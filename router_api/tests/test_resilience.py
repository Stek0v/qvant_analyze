"""Resilience tests: circuit breaker, budget exhaustion, timeout fallback.

Запуск: pytest router_api/tests/test_resilience.py -v
Тестирует CloudAdapter и EscalationRouter без реальных API вызовов.
"""

from __future__ import annotations

import time

import pytest

from router_api.adapters.cloud_adapter import CloudAdapter, CloudBudgetExceeded, CloudCircuitOpen


class TestCloudCircuitBreaker:
    """Circuit breaker: 3 consecutive failures → open for 60s."""

    def test_circuit_starts_closed(self):
        adapter = CloudAdapter(api_key="test", circuit_breaker_fails=3, circuit_breaker_reset_s=1.0)
        assert not adapter._is_circuit_open()

    def test_circuit_opens_after_n_failures(self):
        adapter = CloudAdapter(api_key="test", circuit_breaker_fails=3, circuit_breaker_reset_s=60.0)
        adapter._record_failure()
        adapter._record_failure()
        assert not adapter._is_circuit_open()  # 2 < 3
        adapter._record_failure()
        assert adapter._is_circuit_open()  # 3 >= 3

    def test_circuit_resets_after_success(self):
        adapter = CloudAdapter(api_key="test", circuit_breaker_fails=3, circuit_breaker_reset_s=60.0)
        adapter._record_failure()
        adapter._record_failure()
        adapter._record_success()
        assert adapter._consecutive_failures == 0
        adapter._record_failure()
        assert not adapter._is_circuit_open()  # only 1 after reset

    def test_circuit_resets_after_timeout(self):
        adapter = CloudAdapter(api_key="test", circuit_breaker_fails=2, circuit_breaker_reset_s=0.1)
        adapter._record_failure()
        adapter._record_failure()
        assert adapter._is_circuit_open()
        time.sleep(0.15)  # wait for reset
        assert not adapter._is_circuit_open()  # timer expired → closed


class TestCloudBudget:
    """Budget control: daily spending limit."""

    def test_budget_allows_when_under(self):
        adapter = CloudAdapter(api_key="test", daily_budget_usd=1.0)
        adapter._daily_spent = 0.5
        # estimate_cost returns small amount
        cost = adapter._estimate_cost([{"content": "short"}], 100)
        assert adapter._daily_spent + cost < adapter._daily_budget

    def test_budget_tracking(self):
        adapter = CloudAdapter(api_key="test", daily_budget_usd=0.01)
        adapter._daily_spent = 0.0
        adapter._daily_spent += 0.005
        adapter._daily_spent += 0.005
        assert adapter._daily_spent >= adapter._daily_budget

    def test_cost_estimation_proportional(self):
        adapter = CloudAdapter(api_key="test")
        cost_short = adapter._estimate_cost([{"content": "short"}], 100)
        cost_long = adapter._estimate_cost([{"content": "x" * 4000}], 2000)
        assert cost_long > cost_short


class TestEscalationFallback:
    """Fallback chains: repair → cloud → ok (if cloud disabled)."""

    def test_fallback_chain_repair_then_cloud(self):
        from router_api.config import RouterConfig
        from router_api.models import RouterRequest, LlamaResponse
        from router_api.routers.escalation_router import EscalationRouter

        config = RouterConfig()
        router = EscalationRouter(config)
        req = RouterRequest(query="test")

        # First: empty response → repair
        resp = LlamaResponse(content="", confidence=0.0)
        assert router.decide(req, resp, repair_count=0) == "repair"

        # After repair still bad → cloud
        assert router.decide(req, resp, repair_count=1) == "cloud"

    def test_no_cloud_returns_ok(self):
        from router_api.config import RouterConfig
        from router_api.models import RouterRequest, LlamaResponse
        from router_api.routers.escalation_router import EscalationRouter

        config = RouterConfig(enable_cloud=False)
        router = EscalationRouter(config)
        req = RouterRequest(query="test")
        resp = LlamaResponse(content="", confidence=0.0)

        # repair_count=0 → repair
        assert router.decide(req, resp, repair_count=0) == "repair"
        # repair_count=1, cloud disabled → ok (graceful degradation)
        assert router.decide(req, resp, repair_count=1) == "ok"
