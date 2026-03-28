"""Semantic Pipeline Router (Iteration 2). ARCH-12.

Lightweight semantic router using llama-server embeddings.
No dependency on semantic-router library (incompatible with Python 3.14).
Uses cosine similarity between query embedding and route centroids.

Hard rules (risk_level=high → cloud) override semantic decision.
"""

from __future__ import annotations

import logging
import time

import httpx
import numpy as np

from router_api.config import RouterConfig
from router_api.models import (
    RouterRequest,
    RouterDecision,
    RouteType,
    RagMode,
    RiskLevel,
)

logger = logging.getLogger(__name__)

# Route utterances — used to compute centroids at init time
ROUTE_UTTERANCES: dict[str, list[str]] = {
    "direct_local": [
        "Что такое ACID в базах данных?",
        "Объясни разницу между TCP и UDP",
        "What is the time complexity of binary search?",
        "Напиши функцию сортировки на Python",
        "Explain the CAP theorem",
        "Какие HTTP-коды означают ошибку сервера?",
        "Write a SQL query to find duplicates",
        "Что такое Docker и для чего он нужен?",
        "Объясни принципы SOLID",
        "What is a race condition?",
        "2+2=?",
        "Переведи текст на английский",
        "How does garbage collection work?",
        "Explain microservices vs monolith",
        "Напиши регулярное выражение для email",
    ],
    "local_with_rag": [
        "Что написано в нашей базе знаний про аутентификацию?",
        "Найди в документации описание API эндпоинта",
        "Какие правила деплоя у нас в проекте?",
        "По документу, какие зависимости у сервиса?",
        "What does our internal style guide say?",
        "Найди в базе информацию о конфигурации",
        "Что у нас написано про процесс код-ревью?",
        "Какие тесты покрывают этот модуль?",
        "Из нашей базы: какие метрики мы отслеживаем?",
        "Какая архитектура у нашего сервиса?",
        "Расскажи про наш CI/CD пайплайн из документации",
        "Найди описание ролей и прав доступа",
        "What are the SLA requirements in our project?",
        "Какие ограничения на запросы к внешним API?",
        "Find internal documentation about migrations",
    ],
    "local_with_rag_deep": [
        "Сравни версии конфигурации до и после миграции",
        "Найди расхождения между документацией и кодом",
        "Сопоставь требования из ТЗ с реализацией",
        "По всем документам: какие сервисы зависят от Redis?",
        "Найди противоречия в гайдлайнах по безопасности",
        "Сделай итог по всем материалам о производительности",
        "Compare old and new authentication approaches in docs",
        "Собери информацию из всех источников об инфраструктуре",
        "Find all mentions of deprecated APIs across docs",
        "Суммаризируй все решения по архитектуре",
        "Сравни описание процесса в разных гайдлайнах",
        "Проанализируй все документы связанные с GDPR",
        "Какие конфликтующие рекомендации есть в гайдах?",
    ],
    "cloud_candidate": [
        "Составь юридическое заключение по трудовому спору",
        "Рассчитай налоговые обязательства для ООО",
        "Проведи аудит безопасности данного кода",
        "Draft a legally binding non-disclosure agreement",
        "Calculate the exact financial exposure for this contract",
        "Подготовь отчёт для регулятора по комплаенсу",
        "Проверь соответствие GDPR для схемы обработки данных",
        "Составь точный расчёт пени за просрочку",
        "Assess the legal implications of this data breach",
        "Подготовь экспертное заключение для суда",
        "Review medical protocol for contraindications",
        "Evaluate regulatory compliance for data transfer",
    ],
}

# Map route name → (RouteType, RagMode)
ROUTE_MAP = {
    "direct_local": (RouteType.DIRECT_LOCAL, RagMode.NONE),
    "local_with_rag": (RouteType.LOCAL_WITH_RAG, RagMode.SHALLOW),
    "local_with_rag_deep": (RouteType.LOCAL_WITH_RAG, RagMode.DEEP),
    "cloud_candidate": (RouteType.CLOUD_ESCALATION, RagMode.NONE),
}


class SemanticPipelineRouter:
    def __init__(self, config: RouterConfig):
        self._config = config
        self._centroids: dict[str, np.ndarray] = {}
        self._embedding_url = config.llama_url
        self._rules_fallback = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy init: compute centroids from utterances on first call."""
        if self._initialized:
            return
        try:
            self._compute_centroids()
            self._initialized = True
            logger.info(f"Semantic Router initialized with {len(self._centroids)} routes")
        except Exception as e:
            logger.warning(f"Semantic Router init failed: {e}. Using rules fallback.")

    def _get_embedding(self, texts: list[str]) -> np.ndarray:
        """Получить embeddings через llama-server /v1/embeddings."""
        r = httpx.post(
            f"{self._embedding_url}/v1/embeddings",
            json={"input": texts, "model": "qwen3.5-27b"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return np.array([d["embedding"] for d in data["data"]])

    def _compute_centroids(self):
        """Вычислить centroid для каждого route из utterances."""
        for route_name, utterances in ROUTE_UTTERANCES.items():
            embeddings = self._get_embedding(utterances)
            centroid = embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid)  # normalize
            self._centroids[route_name] = centroid

    def _classify(self, query: str) -> tuple[str, float]:
        """Classify query by cosine similarity to centroids."""
        query_emb = self._get_embedding([query])[0]
        query_emb /= np.linalg.norm(query_emb)

        best_route = "direct_local"
        best_score = -1.0

        for route_name, centroid in self._centroids.items():
            score = float(np.dot(query_emb, centroid))
            if score > best_score:
                best_score = score
                best_route = route_name

        return best_route, best_score

    def route(self, request: RouterRequest) -> RouterDecision:
        reasons: list[str] = []

        # Hard rule: high risk → cloud (always)
        if request.risk_level == RiskLevel.HIGH:
            reasons.append("high_risk")
            return RouterDecision(
                route=RouteType.CLOUD_ESCALATION,
                rag_mode=RagMode.NONE,
                cloud_allowed=True,
                reason_codes=reasons,
            )

        # Hard rule: require_json → structured path
        if request.require_json:
            reasons.append("format_strict")
            return RouterDecision(
                route=RouteType.LOCAL_WITH_RAG_AND_REPAIR,
                rag_mode=RagMode.SHALLOW,
                cloud_allowed=True,
                reason_codes=reasons,
            )

        # Semantic classification
        self._ensure_initialized()

        if self._centroids:
            try:
                route_name, score = self._classify(request.query)
                route_type, rag_mode = ROUTE_MAP.get(
                    route_name, (RouteType.DIRECT_LOCAL, RagMode.NONE)
                )
                reasons.append(f"semantic_{route_name}")
                return RouterDecision(
                    route=route_type,
                    rag_mode=rag_mode,
                    cloud_allowed=True,
                    reason_codes=reasons,
                )
            except Exception as e:
                logger.warning(f"Semantic classify error: {e}")

        # Fallback to rules
        if not self._rules_fallback:
            from router_api.routers.pipeline_router import PipelineRouter
            self._rules_fallback = PipelineRouter(self._config)
        return self._rules_fallback.route(request)
