"""Block II: Route quality tests. QA-02.

Проверяет правильность маршрутизации на реальных примерах.
Считает 5 метрик из v2+routes.md.

Запуск: pytest router_api/tests/test_route_quality.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8100"
TIMEOUT = 120.0
GOLDEN_PATH = Path(__file__).parent.parent / "golden_dataset.json"


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
        yield c


def _route(client, query, **kwargs):
    payload = {"query": query, **kwargs}
    r = client.post("/v1/route", json=payload,
                    headers={"Content-Type": "application/json", "X-Debug": "true"})
    assert r.status_code == 200
    return r.json()


# ── Individual route tests ──

def test_01_simple_factual_goes_direct(client):
    """Простой вопрос → direct_local."""
    d = _route(client, "Что такое HTTP протокол?")
    assert d["route"] == "direct_local"


def test_02_internal_goes_rag(client):
    """Внутренний вопрос → local_with_rag."""
    d = _route(client, "Что написано в нашей базе знаний?", source_hint="internal")
    assert d["route"] in ("local_with_rag", "local_with_rag_and_repair")


def test_03_general_not_rag(client):
    """Общий вопрос НЕ идёт в RAG."""
    d = _route(client, "Explain the difference between TCP and UDP")
    assert d["route"] == "direct_local", f"General query sent to RAG: {d['route']}"


def test_04_general_not_cloud(client):
    """Общий вопрос НЕ идёт в cloud."""
    d = _route(client, "Что такое ACID?")
    assert d["route"] != "cloud_escalation", f"General query sent to cloud: {d['route']}"


def test_05_high_risk_strict(client):
    """High-risk → cloud_escalation."""
    d = _route(client, "Критически важный запрос", risk_level="high")
    assert d["route"] == "cloud_escalation"


def test_06_ambiguous_not_cloud(client):
    """Неоднозначный запрос НЕ сразу в cloud."""
    d = _route(client, "Расскажи про Docker и как мы его используем")
    assert d["route"] != "cloud_escalation"


def test_07_deep_rag_keywords(client):
    """Multi-doc вопрос → deep RAG."""
    d = _route(client, "Сравни все версии конфигурации и найди расхождения", source_hint="internal")
    assert d["rag_mode"] == "deep" or d["route"] == "local_with_rag"


def test_08_json_structured(client):
    """require_json → structured path."""
    d = _route(client, "Извлеки данные", require_json=True)
    assert d["route"] == "local_with_rag_and_repair"


# ── Aggregate metrics on golden dataset subset ──

def test_09_route_accuracy_above_threshold(client):
    """Route accuracy на первых 30 кейсах ≥ 75%."""
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)[:30]

    correct = 0
    for case in golden:
        d = _route(client, case["query"],
                   risk_level=case.get("risk_level", "low"),
                   source_hint=case.get("source_hint", "none"),
                   require_json=case.get("require_json", False))
        if d.get("route") == case["expected_route"]:
            correct += 1

    accuracy = correct / len(golden)
    assert accuracy >= 0.75, f"Route accuracy {accuracy:.1%} < 75%"


def test_10_unnecessary_cloud_below_10pct(client):
    """Unnecessary cloud rate < 10% на первых 30 кейсах."""
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)[:30]

    unnecessary = 0
    for case in golden:
        d = _route(client, case["query"],
                   risk_level=case.get("risk_level", "low"),
                   source_hint=case.get("source_hint", "none"))
        if case["expected_route"] != "cloud_escalation" and d.get("route") == "cloud_escalation":
            unnecessary += 1

    rate = unnecessary / len(golden)
    assert rate < 0.10, f"Unnecessary cloud rate {rate:.1%} >= 10%"
