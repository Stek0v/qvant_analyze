"""Конфигурация Router API (YAML + env vars)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class RoutingThresholds(BaseSettings):
    """Пороги маршрутизации из v2+routes.md (строки 612-619)."""
    rag_shallow_min_score: float = 0.78
    rag_shallow_min_gap: float = 0.08
    rag_deep_min_score: float = 0.60
    rag_deep_max_score: float = 0.78


class RouterConfig(BaseSettings):
    """Центральная конфигурация Router API."""

    model_config = {"env_prefix": "ROUTER_"}

    # Endpoints
    llama_url: str = "http://127.0.0.1:11434"
    cognee_url: str = "http://127.0.0.1:9000"
    langfuse_host: str = "http://127.0.0.1:3000"
    redis_url: str = "redis://127.0.0.1:6380/0"

    # Cloud
    cloud_api_url: str = "https://openrouter.ai/api/v1"
    cloud_model: str = "anthropic/claude-sonnet-4"
    cloud_daily_budget_usd: float = 5.0
    cloud_timeout_s: float = 60.0
    cloud_circuit_breaker_fails: int = 3
    cloud_circuit_breaker_reset_s: float = 60.0

    # Inference
    llama_timeout_s: float = 300.0
    llama_max_tokens: int = 2048

    # Routing
    thresholds: RoutingThresholds = Field(default_factory=RoutingThresholds)

    # RAG
    rag_shallow_top_k: int = 5
    rag_deep_top_k: int = 20

    # Feature flags
    enable_rag: bool = True
    enable_cloud: bool = True
    enable_semantic_router: bool = False  # Iter 2
    enable_verifier: bool = False  # Iter 2

    # Security
    api_keys: list[str] = Field(default_factory=list)
    rate_limit_per_min: int = 60

    # Keywords для Pipeline Router (русские маркеры внутренних запросов)
    rag_keywords: list[str] = Field(default_factory=lambda: [
        "по документу", "в базе", "в проекте",
        "найди расхождения", "сравни версии",
        "что у нас написано", "из нашей базы",
        "внутренний", "наш проект",
    ])

    # Paths
    config_dir: Path = Path(__file__).parent


def load_config() -> RouterConfig:
    """Загрузить конфигурацию из env vars."""
    return RouterConfig()
