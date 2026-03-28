"""Async adapter для llama-server. ARCH-06.

Обёртка поверх /v1/chat/completions с JSON mode, repair и timeout handling.
"""

from __future__ import annotations

import time

import httpx

from router_api.models import LlamaResponse


class BackendTimeoutError(Exception):
    pass


class LlamaAdapter:
    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 300.0, max_tokens: int = 2048):
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        )

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health")
            return r.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: dict | None = None,
    ) -> LlamaResponse:
        """Генерация ответа через llama-server.

        Args:
            messages: OpenAI-формат сообщений
            temperature: Температура генерации
            max_tokens: Лимит токенов (None = из конфига)
            json_schema: JSON schema для structured output (None = свободный текст)
        """
        payload: dict = {
            "model": "qwen3.5-27b",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }

        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }

        t0 = time.monotonic()
        try:
            r = await self._client.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(f"llama-server timeout after {time.monotonic() - t0:.1f}s") from e
        except httpx.HTTPStatusError as e:
            return LlamaResponse(content="", confidence=0.0, latency_ms=(time.monotonic() - t0) * 1000)

        latency_ms = (time.monotonic() - t0) * 1000

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        timings = data.get("timings", {})
        usage = data.get("usage", {})

        return LlamaResponse(
            content=msg.get("content", "") or "",
            thinking=msg.get("reasoning_content"),
            confidence=0.7,  # TODO: extract from structured output
            tokens_per_sec=timings.get("predicted_per_second", 0),
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def generate_with_context(
        self,
        query: str,
        context: list[str] | None = None,
        history_summary: str | None = None,
        require_json: bool = False,
        repair: bool = False,
        json_schema: dict | None = None,
    ) -> LlamaResponse:
        """Высокоуровневый метод: собрать messages и сгенерировать."""
        system = "Ты отвечаешь ясно и по делу."

        if context:
            ctx_text = "\n\n---\n\n".join(context)
            system += f"\n\nКонтекст из базы знаний:\n{ctx_text}"

        if repair:
            system += "\nПредыдущий ответ был неудачным. Дай более точный и полный ответ."

        if require_json:
            system += "\nОтвечай строго в формате JSON."

        messages = [{"role": "system", "content": system}]

        if history_summary:
            messages.append({"role": "system", "content": f"Краткое резюме диалога: {history_summary}"})

        messages.append({"role": "user", "content": query})

        return await self.generate(messages, json_schema=json_schema if require_json else None)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
