"""Автоматическая и ручная оценка качества ответов."""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import TOKEN_TARGETS
from test_prompts import SET_A_GROUND_TRUTH, SET_B_EXPECTED_SECTIONS, SET_C_EXPECTED_SECTIONS


@dataclass
class QualityScores:
    completeness: float = 0.0   # 0-100
    structure: float = 0.0      # 0-100
    conciseness: float = 0.0    # 0-100
    composite: float = 0.0      # 0-100


def score_response(content: str, prompt_set: str, prompt_index: int,
                   thinking: str | None = None) -> QualityScores:
    """Автоматически оценить ответ."""
    completeness = _score_completeness(content, prompt_set, prompt_index)
    structure = _score_structure(content)
    conciseness = _score_conciseness(content, prompt_set)

    # Взвешенный композит: completeness важнее всего
    composite = completeness * 0.5 + structure * 0.25 + conciseness * 0.25

    return QualityScores(
        completeness=round(completeness, 1),
        structure=round(structure, 1),
        conciseness=round(conciseness, 1),
        composite=round(composite, 1),
    )


# ------------------------------------------------------------------
# Completeness
# ------------------------------------------------------------------
def _score_completeness(content: str, prompt_set: str, prompt_index: int) -> float:
    content_lower = content.lower()

    if prompt_set == "A":
        gt = SET_A_GROUND_TRUTH.get(prompt_index, {})
        return _check_ground_truth(content_lower, gt)

    elif prompt_set == "B":
        sections = SET_B_EXPECTED_SECTIONS.get(prompt_index, [])
        return _check_expected_sections(content_lower, sections)

    elif prompt_set == "C":
        sections = SET_C_EXPECTED_SECTIONS.get(prompt_index, [])
        return _check_expected_sections(content_lower, sections)

    return 50.0


def _check_ground_truth(content_lower: str, gt: dict) -> float:
    if not gt:
        return 50.0

    scores = []

    # Проверка обязательных ключевых слов
    keywords = gt.get("required_keywords", [])
    if keywords:
        # Считаем сколько из альтернативных вариантов присутствует
        # Группируем по семантике: если любой из вариантов найден — засчитываем
        found = sum(1 for kw in keywords if kw.lower() in content_lower)
        # Для ACID нужно минимум 4 из 8 (4 англ + 4 рус)
        ratio = min(found / max(len(keywords) // 2, 1), 1.0)
        scores.append(ratio * 100)

    # Проверка точных ответов
    exact = gt.get("exact_answers", [])
    if exact:
        found = sum(1 for a in exact if a in content_lower)
        scores.append(found / len(exact) * 100)

    # Проверка точного вывода
    exact_out = gt.get("exact_output")
    if exact_out:
        scores.append(100.0 if exact_out in content_lower else 0.0)

    # Минимум элементов
    min_items = gt.get("min_items")
    if min_items:
        # Считаем буллеты, нумерацию или переводы строк
        items = len(re.findall(r"^[\-\*\d]+[\.\)]\s", content_lower, re.MULTILINE))
        if items == 0:
            items = content_lower.count("\n")
        scores.append(min(items / min_items, 1.0) * 100)

    return sum(scores) / len(scores) if scores else 50.0


def _check_expected_sections(content_lower: str, sections: list[str]) -> float:
    if not sections:
        return 50.0
    found = sum(1 for s in sections if s.lower() in content_lower)
    return found / len(sections) * 100


# ------------------------------------------------------------------
# Structure
# ------------------------------------------------------------------
def _score_structure(content: str) -> float:
    """Оценка структурированности: заголовки, списки, блоки кода."""
    if not content.strip():
        return 0.0

    lines = content.split("\n")
    total = len(lines)
    if total == 0:
        return 0.0

    headers = sum(1 for l in lines if re.match(r"^#{1,4}\s", l))
    bullets = sum(1 for l in lines if re.match(r"^\s*[\-\*]\s", l))
    numbered = sum(1 for l in lines if re.match(r"^\s*\d+[\.\)]\s", l))
    code_blocks = content.count("```")

    structure_elements = headers + bullets + numbered + code_blocks // 2

    if total < 5:
        # Короткий ответ — структура не обязательна
        return 70.0 if structure_elements > 0 else 50.0

    # Для длинных ответов ожидаем структуру
    ratio = structure_elements / total
    if ratio > 0.3:
        return 100.0
    elif ratio > 0.15:
        return 80.0
    elif ratio > 0.05:
        return 60.0
    elif structure_elements > 0:
        return 40.0
    return 20.0


# ------------------------------------------------------------------
# Conciseness
# ------------------------------------------------------------------
def _score_conciseness(content: str, prompt_set: str) -> float:
    """Оценка лаконичности: штраф за слишком короткий или длинный ответ."""
    tokens = len(content.split())
    min_t, max_t = TOKEN_TARGETS.get(prompt_set, (100, 500))

    if min_t <= tokens <= max_t:
        return 100.0
    elif tokens < min_t:
        # Слишком короткий
        return max(0, tokens / min_t * 100)
    else:
        # Слишком длинный — мягкий штраф
        overshoot = tokens / max_t
        if overshoot < 1.5:
            return 80.0
        elif overshoot < 2.0:
            return 60.0
        elif overshoot < 3.0:
            return 40.0
        return 20.0


# ------------------------------------------------------------------
# Ручная оценка (CLI)
# ------------------------------------------------------------------
def manual_scoring_session(results: list[dict]) -> None:
    """Интерактивная сессия ручной оценки через rich."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print("Установите rich: pip install rich")
        return

    console = Console()

    for r in results:
        if r.get("manual_accuracy") is not None:
            continue  # Уже оценён

        console.print(Panel(r["prompt_text"][:200], title="Промпт", border_style="blue"))
        console.print(Panel(r["content"][:1000], title=f"Ответ [{r['config_name']}]", border_style="green"))

        if r.get("thinking"):
            console.print(Panel(r["thinking"][:500], title="Thinking", border_style="yellow"))

        table = Table(title="Автоматические оценки")
        table.add_column("Метрика")
        table.add_column("Балл")
        table.add_row("Completeness", str(r.get("quality_completeness", "?")))
        table.add_row("Structure", str(r.get("quality_structure", "?")))
        table.add_row("Conciseness", str(r.get("quality_conciseness", "?")))
        table.add_row("Composite", str(r.get("quality_composite", "?")))
        console.print(table)

        console.print("\n[bold]Ручная оценка (1-5, Enter=пропустить, q=выход):[/bold]")
        for field, label in [
            ("manual_accuracy", "Точность"),
            ("manual_completeness", "Полнота"),
            ("manual_reasoning", "Качество рассуждений"),
            ("manual_usefulness", "Практическая полезность"),
        ]:
            val = input(f"  {label}: ").strip()
            if val == "q":
                return
            if val.isdigit() and 1 <= int(val) <= 5:
                r[field] = int(val)

        console.print("---\n")
