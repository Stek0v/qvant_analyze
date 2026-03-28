"""Router 2 — RAG Router (threshold-based). ARCH-04.

Решает: none | shallow | deep | requery на основе retrieval сигналов.
"""

from __future__ import annotations

from router_api.config import RouterConfig
from router_api.models import RagMode, RetrievalResult


class RagDecision:
    def __init__(self, mode: RagMode, reason: str):
        self.mode = mode
        self.reason = reason


class RAGRouter:
    def __init__(self, config: RouterConfig):
        self._cfg = config.thresholds

    def decide(self, results: list[RetrievalResult]) -> RagDecision:
        if not results:
            return RagDecision(RagMode.NONE, "empty_results")

        scores = sorted([r.score for r in results], reverse=True)
        top1 = scores[0]
        gap = (top1 - scores[1]) if len(scores) > 1 else top1

        # Высокий score + большой gap → shallow достаточно
        if top1 >= self._cfg.rag_shallow_min_score and gap >= self._cfg.rag_shallow_min_gap:
            return RagDecision(RagMode.SHALLOW, "retrieval_high_confidence")

        # Средний score → deep для лучшего контекста
        if top1 >= self._cfg.rag_deep_min_score:
            return RagDecision(RagMode.DEEP, "retrieval_low_confidence")

        # Очень низкий score → retrieval бесполезен
        return RagDecision(RagMode.NONE, "retrieval_low_confidence")
