"""Обёртка над llama-server / Ollama API для бенчмаркинга.

Поддерживает оба backend-а:
- llama-server: OpenAI-совместимый /v1/chat/completions, /health
- Ollama: /api/chat, /api/ps, /api/show
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import httpx

from config import (
    OLLAMA_BASE_URL,
    MODEL_DEFAULT,
    TEST_TIMEOUT_SECONDS,
    HEALTH_CHECK_RETRIES,
    HEALTH_CHECK_INTERVAL,
    BenchmarkResult,
)


def detect_backend(base_url: str) -> str:
    """Определить backend: 'llama' или 'ollama'."""
    try:
        r = httpx.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            return "llama"
    except Exception:
        pass
    try:
        r = httpx.get(f"{base_url}/", timeout=3)
        if r.status_code == 200:
            return "ollama"
    except Exception:
        pass
    return "unknown"


class InferenceClient:
    """Универсальный клиент для llama-server и Ollama."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, backend: str | None = None,
                 model_vram_mb: int | None = None):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=TEST_TIMEOUT_SECONDS)
        self.backend = backend or detect_backend(self.base_url)
        self._model_vram_mb = model_vram_mb

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def wait_ready(self) -> bool:
        """Дождаться доступности сервера."""
        for _ in range(HEALTH_CHECK_RETRIES):
            try:
                if self.backend == "llama":
                    r = self._http.get("/health")
                else:
                    r = self._http.get("/")
                if r.status_code == 200:
                    return True
            except httpx.ConnectError:
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)
        return False

    # ------------------------------------------------------------------
    # GPU info
    # ------------------------------------------------------------------
    def get_gpu_percent(self) -> str:
        """Вернуть строку GPU%. Для llama-server — через nvidia-smi."""
        if self.backend == "ollama":
            return self._ollama_gpu_percent()
        # llama-server не имеет /api/ps — используем nvidia-smi через gpu_monitor
        from gpu_monitor import snapshot
        try:
            gpu = snapshot()
            # Динамический порог: модель в GPU если VRAM > 85% размера модели
            threshold = int(self._model_vram_mb * 0.85) if self._model_vram_mb else 2000
            if gpu.vram_used_mb > threshold:
                return "100% GPU"
            return f"~{gpu.vram_used_mb} MB used"
        except Exception:
            return "unknown"

    def _ollama_gpu_percent(self) -> str:
        try:
            r = self._http.get("/api/ps")
            r.raise_for_status()
            models = r.json().get("models", [])
            if not models:
                return "no model loaded"
            info = models[0]
            size = info.get("size", 1)
            size_vram = info.get("size_vram", 0)
            if size and size_vram:
                gpu_pct = round(size_vram / size * 100)
                if gpu_pct >= 99:
                    return "100% GPU"
                return f"{100 - gpu_pct}%/{gpu_pct}% CPU/GPU"
        except Exception:
            pass
        return "unknown"

    # ------------------------------------------------------------------
    # Chat — основной метод
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        model: str = MODEL_DEFAULT,
        think: bool | str = False,
        num_ctx: int = 8192,
        temperature: float = 0.2,
        top_p: float = 0.95,
        presence_penalty: float | None = None,
        max_tokens: int = 2048,
        keep_alive: str = "30m",
    ) -> BenchmarkResult:
        if self.backend == "llama":
            return self._chat_llama(
                messages, model=model, think=think, temperature=temperature,
                top_p=top_p, presence_penalty=presence_penalty, max_tokens=max_tokens,
            )
        return self._chat_ollama(
            messages, model=model, think=think, num_ctx=num_ctx,
            temperature=temperature, top_p=top_p,
            presence_penalty=presence_penalty, keep_alive=keep_alive,
        )

    # ------------------------------------------------------------------
    # llama-server: /v1/chat/completions
    # ------------------------------------------------------------------
    def _chat_llama(
        self,
        messages: list[dict],
        *,
        model: str,
        think: bool | str,
        temperature: float,
        top_p: float,
        presence_penalty: float | None,
        max_tokens: int,
    ) -> BenchmarkResult:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty

        ts = datetime.now(timezone.utc).isoformat()
        r = self._http.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()

        # Разбор ответа
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        thinking = msg.get("reasoning_content") or None

        # Метрики из timings
        timings = data.get("timings", {})
        usage = data.get("usage", {})

        prompt_count = timings.get("prompt_n", usage.get("prompt_tokens", 0))
        eval_count = timings.get("predicted_n", usage.get("completion_tokens", 0))
        prompt_ms = timings.get("prompt_ms", 0)
        eval_ms = timings.get("predicted_ms", 0)
        tok_s = timings.get("predicted_per_second", 0)
        prompt_tok_s = timings.get("prompt_per_second", 0)
        total_ms = prompt_ms + eval_ms

        # Грубая оценка токенов
        content_tokens = len(content.split()) if content else 0
        thinking_tokens = len(thinking.split()) if thinking else 0

        return BenchmarkResult(
            run_id=str(uuid.uuid4()),
            timestamp=ts,
            model=model,
            num_ctx=0,  # llama-server не переключает ctx per-request
            think_mode=think,
            temperature=temperature,
            total_duration_ms=total_ms,
            load_duration_ms=0,
            prompt_eval_count=prompt_count,
            prompt_eval_duration_ms=prompt_ms,
            eval_count=eval_count,
            eval_duration_ms=eval_ms,
            tokens_per_sec=tok_s,
            prompt_tokens_per_sec=prompt_tok_s,
            is_cold_start=False,
            content=content,
            thinking=thinking,
            content_token_estimate=content_tokens,
            thinking_token_estimate=thinking_tokens,
        )

    # ------------------------------------------------------------------
    # Ollama: /api/chat
    # ------------------------------------------------------------------
    def _chat_ollama(
        self,
        messages: list[dict],
        *,
        model: str,
        think: bool | str,
        num_ctx: int,
        temperature: float,
        top_p: float,
        presence_penalty: float | None,
        keep_alive: str,
    ) -> BenchmarkResult:
        payload: dict = {
            "model": model,
            "messages": messages,
            "think": think,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        if presence_penalty is not None:
            payload["options"]["presence_penalty"] = presence_penalty

        ts = datetime.now(timezone.utc).isoformat()
        r = self._http.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

        msg = data.get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking")

        total_ns = data.get("total_duration", 0)
        load_ns = data.get("load_duration", 0)
        prompt_eval_ns = data.get("prompt_eval_duration", 0)
        eval_ns = data.get("eval_duration", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        total_ms = total_ns / 1e6
        load_ms = load_ns / 1e6
        prompt_eval_ms = prompt_eval_ns / 1e6
        eval_ms = eval_ns / 1e6

        tok_s = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
        prompt_tok_s = (prompt_eval_count / (prompt_eval_ns / 1e9)) if prompt_eval_ns > 0 else 0.0
        is_cold = load_ns > (total_ns * 0.5) if total_ns > 0 else False

        content_tokens = len(content.split())
        thinking_tokens = len(thinking.split()) if thinking else 0

        return BenchmarkResult(
            run_id=str(uuid.uuid4()),
            timestamp=ts,
            model=model,
            num_ctx=num_ctx,
            think_mode=think,
            temperature=temperature,
            total_duration_ms=total_ms,
            load_duration_ms=load_ms,
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration_ms=prompt_eval_ms,
            eval_count=eval_count,
            eval_duration_ms=eval_ms,
            tokens_per_sec=tok_s,
            prompt_tokens_per_sec=prompt_tok_s,
            is_cold_start=is_cold,
            content=content,
            thinking=thinking,
            content_token_estimate=content_tokens,
            thinking_token_estimate=thinking_tokens,
        )

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------
    def unload(self, model: str = MODEL_DEFAULT) -> None:
        """Выгрузить модель (только Ollama)."""
        if self.backend == "ollama":
            self._http.post("/api/chat", json={
                "model": model, "messages": [], "keep_alive": "0", "stream": False,
            })

    def warmup(self, model: str = MODEL_DEFAULT, num_ctx: int = 8192) -> BenchmarkResult:
        """Прогревочный запрос."""
        return self.chat(
            messages=[{"role": "user", "content": "Привет, ответь одним словом."}],
            model=model, think=False, num_ctx=num_ctx,
            temperature=0.1, max_tokens=50,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Обратная совместимость
OllamaClient = InferenceClient
