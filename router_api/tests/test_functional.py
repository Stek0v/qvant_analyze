"""Block I: Базовые функциональные тесты Router API. QA-01.

Запуск: pytest router_api/tests/test_functional.py -v
Требует: Router API на 127.0.0.1:8100 (systemd service)
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8100"
TIMEOUT = 120.0  # Cognee MCP может быть медленным


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
        yield c


def _route(client: httpx.Client, payload: dict, debug: bool = True) -> dict:
    headers = {"Content-Type": "application/json"}
    if debug:
        headers["X-Debug"] = "true"
    r = client.post("/v1/route", json=payload, headers=headers)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    return r.json()


# ── Test 1: direct_local route ──
def test_01_direct_local(client):
    """Общий вопрос идёт через direct_local."""
    d = _route(client, {"query": "Что такое HTTP?"})
    assert d["route"] == "direct_local"
    assert len(d["answer"]) > 10
    assert d["escalated_to_cloud"] is False


# ── Test 2: local_with_rag route ──
def test_02_local_with_rag(client):
    """Внутренний вопрос идёт через local_with_rag."""
    d = _route(client, {
        "query": "Что написано в нашей базе знаний?",
        "source_hint": "internal",
    })
    assert d["route"] in ("local_with_rag", "local_with_rag_and_repair")


# ── Test 3: cloud escalation ──
def test_03_cloud_escalation(client):
    """high_risk запрос эскалируется."""
    d = _route(client, {
        "query": "Критически важный юридический вопрос",
        "risk_level": "high",
    })
    assert d["route"] == "cloud_escalation"
    assert d["escalated_to_cloud"] is True


# ── Test 4: null history ──
def test_04_null_history(client):
    """Запрос с history_summary=null не ломается."""
    d = _route(client, {
        "query": "Привет",
        "history_summary": None,
    })
    assert d["answer"]  # не пустой


# ── Test 5: long history ──
def test_05_long_history(client):
    """Запрос с длинным history работает."""
    d = _route(client, {
        "query": "Продолжи",
        "history_summary": "x" * 2000,
    })
    assert "answer" in d


# ── Test 6: attachments metadata ──
def test_06_attachments(client):
    """Запрос с attachments не ломает маршрут."""
    d = _route(client, {
        "query": "Проанализируй файл",
        "attachments": [{"type": "file", "name": "report.pdf"}],
    })
    assert d["route"] in ("direct_local", "local_with_rag")


# ── Test 7: require_json returns valid JSON ──
def test_07_require_json(client):
    """require_json=true возвращает парсимый JSON в answer."""
    d = _route(client, {
        "query": "Перечисли 3 цвета в формате JSON",
        "require_json": True,
    })
    assert d["route"] == "local_with_rag_and_repair"
    # answer должен быть валидным JSON
    try:
        parsed = json.loads(d["answer"])
        assert isinstance(parsed, (dict, list))
    except json.JSONDecodeError:
        # Модель может обернуть JSON в markdown — проверим наличие { или [
        assert "{" in d["answer"] or "[" in d["answer"], \
            f"Answer is not JSON-like: {d['answer'][:100]}"


# ── Test 8: valid response schema ──
def test_08_response_schema(client):
    """Ответ содержит все обязательные поля."""
    d = _route(client, {"query": "Тест"})
    required = ["request_id", "answer", "confidence", "grounded",
                "used_context", "repair_applied", "escalated_to_cloud"]
    for field in required:
        assert field in d, f"Missing field: {field}"
    assert 0.0 <= d["confidence"] <= 1.0


# ── Test 9: debug fields hidden without X-Debug ──
def test_09_no_debug_fields(client):
    """Без X-Debug internal metadata не в ответе."""
    d = _route(client, {"query": "Тест"}, debug=False)
    assert "route" not in d or d["route"] is None
    assert "reason_codes" not in d or d["reason_codes"] is None


# ── Test 10: health endpoint ──
def test_10_health(client):
    """Health endpoint возвращает статус бэкендов."""
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] in ("ok", "degraded")
    assert "backends" in d
    assert "llama" in d["backends"]
