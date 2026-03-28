"""Async adapter для cloud escalation через OpenRouter. ARCH-08.

Budget control через Redis, circuit breaker (3 fails → 60s open).
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from router_api.models import CloudResponse

logger = logging.getLogger(__name__)


class CloudBudgetExceeded(Exception):
    pass


class CloudCircuitOpen(Exception):
    pass


class CloudAdapter:
    def __init__(
        self,
        api_url: str = "https://openrouter.ai/api/v1",
        model: str = "anthropic/claude-sonnet-4",
        api_key: str | None = None,
        timeout: float = 60.0,
        daily_budget_usd: float = 5.0,
        circuit_breaker_fails: int = 3,
        circuit_breaker_reset_s: float = 60.0,
    ):
        self._api_url = api_url
        self._model = model
        self._api_key = api_key or self._load_api_key()
        self._timeout = timeout
        self._daily_budget = daily_budget_usd

        # Circuit breaker state (in-memory, resets on restart)
        self._consecutive_failures = 0
        self._cb_max_fails = circuit_breaker_fails
        self._cb_reset_s = circuit_breaker_reset_s
        self._cb_open_until = 0.0

        # Budget tracking (in-memory; for Redis see TODO)
        self._daily_spent = 0.0
        self._budget_reset_time = 0.0

        self._client = httpx.AsyncClient(
            base_url=api_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": "https://github.com/Stek0v/qvant_analyze",
                "X-Title": "qvant Router",
            },
        )

    @staticmethod
    def _load_api_key() -> str:
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            env_path = os.path.expanduser("~/.env")
            try:
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("OPENROUTER_API_KEY="):
                            key = line.split("=", 1)[1].strip().strip('"')
                            break
            except FileNotFoundError:
                pass
        return key

    async def generate(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> CloudResponse:
        """Генерация через cloud API."""
        # Circuit breaker check
        if self._is_circuit_open():
            raise CloudCircuitOpen(
                f"Circuit open after {self._cb_max_fails} failures, reset in {self._cb_open_until - time.monotonic():.0f}s"
            )

        # Budget check
        estimated_cost = self._estimate_cost(messages, max_tokens)
        if self._daily_spent + estimated_cost > self._daily_budget:
            raise CloudBudgetExceeded(
                f"Daily budget {self._daily_budget}$ exceeded (spent: {self._daily_spent:.4f}$)"
            )

        t0 = time.monotonic()
        try:
            r = await self._client.post("/chat/completions", json={
                "model": model or self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            })
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self._record_failure()
            logger.warning(f"Cloud API error: {e}")
            return CloudResponse(content="", cost_usd=0.0, latency_ms=(time.monotonic() - t0) * 1000)

        self._record_success()
        latency_ms = (time.monotonic() - t0) * 1000

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})

        # Cost from OpenRouter response
        cost = 0.0
        if "usage" in data:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            # OpenRouter pricing varies; rough estimate
            cost = (prompt_tokens * 3 + completion_tokens * 15) / 1_000_000
        self._daily_spent += cost

        return CloudResponse(
            content=msg.get("content", ""),
            cost_usd=cost,
            latency_ms=latency_ms,
            model=model or self._model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def _estimate_cost(self, messages: list[dict], max_tokens: int) -> float:
        """Грубая оценка стоимости до отправки."""
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = prompt_chars // 4  # ~4 chars per token
        return (prompt_tokens * 3 + max_tokens * 15) / 1_000_000

    def _is_circuit_open(self) -> bool:
        if self._consecutive_failures >= self._cb_max_fails:
            if time.monotonic() < self._cb_open_until:
                return True
            # Reset after timeout
            self._consecutive_failures = 0
        return False

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._cb_max_fails:
            self._cb_open_until = time.monotonic() + self._cb_reset_s
            logger.warning(f"Cloud circuit breaker OPEN for {self._cb_reset_s}s")

    def _record_success(self):
        self._consecutive_failures = 0

    async def close(self):
        await self._client.aclose()
