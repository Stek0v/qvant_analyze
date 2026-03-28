"""Local verifier module (Iteration 2). ARCH-13.

Быстрая локальная проверка ответа тем же Qwen:
- grounded: ответ опирается на контекст?
- format_ok: формат корректен?
- complete: ответ полный?
- gaps: что упущено?

Добавляет < 2s latency. Включается через enable_verifier feature flag.
"""

from __future__ import annotations

import json
import logging

from router_api.adapters.llama_adapter import LlamaAdapter
from router_api.models import VerificationResult

logger = logging.getLogger(__name__)

VERIFY_PROMPT = """Ты — верификатор ответов. Оцени качество ответа по 4 критериям.

Вопрос: {question}

{context_section}

Ответ: {answer}

Оцени строго в JSON формате:
{{
  "grounded": true/false,      // ответ опирается на предоставленный контекст (если он есть)?
  "format_ok": true/false,     // формат ответа корректен и читаем?
  "complete": true/false,      // ответ полно отвечает на вопрос?
  "gaps": ["..."],             // список упущенных моментов (пустой если всё ОК)
  "confidence": 0.0-1.0        // общая уверенность в качестве ответа
}}

Отвечай ТОЛЬКО JSON, без пояснений."""


class LocalVerifier:
    def __init__(self, llama: LlamaAdapter):
        self._llama = llama

    async def verify(
        self,
        question: str,
        answer: str,
        context: list[str] | None = None,
    ) -> VerificationResult:
        """Проверить ответ локальной моделью."""
        if not answer or len(answer.strip()) < 5:
            return VerificationResult(
                grounded=False, format_ok=False, complete=False,
                gaps=["empty answer"], confidence=0.0,
            )

        context_section = ""
        if context:
            ctx_text = "\n---\n".join(context[:3])  # Max 3 docs
            context_section = f"Контекст из базы знаний:\n{ctx_text}"

        prompt = VERIFY_PROMPT.format(
            question=question[:500],
            context_section=context_section,
            answer=answer[:1000],
        )

        try:
            resp = await self._llama.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            return self._parse_result(resp.content)
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return VerificationResult(confidence=0.5)  # Assume OK on failure

    @staticmethod
    def _parse_result(text: str) -> VerificationResult:
        """Парсить JSON ответ верификатора."""
        # Извлечь JSON из ответа (может быть обёрнут в markdown)
        text = text.strip()
        if "```" in text:
            # Извлечь из code block
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        try:
            data = json.loads(text)
            return VerificationResult(
                grounded=bool(data.get("grounded", True)),
                format_ok=bool(data.get("format_ok", True)),
                complete=bool(data.get("complete", True)),
                gaps=data.get("gaps", []),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.debug(f"Failed to parse verification JSON: {text[:100]}")
            return VerificationResult(confidence=0.5)
