#!/usr/bin/env python3
"""Routing accuracy analysis framework. ANALYST-03.

Сравнивает ожидаемые маршруты (golden dataset) с фактическими решениями роутера.
Выводит: accuracy, confusion matrix, 5 secondary metrics.

Использование:
    python -m router_api.eval.routing_accuracy                    # через Router API
    python -m router_api.eval.routing_accuracy --results results.json  # из файла
    python -m router_api.eval.routing_accuracy --limit 20         # первые 20 кейсов
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

GOLDEN_PATH = Path(__file__).parent.parent / "golden_dataset.json"
ROUTER_URL = "http://127.0.0.1:8100"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "eval"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_through_router(cases: list[dict], url: str = ROUTER_URL,
                       timeout: float = 120.0) -> list[dict]:
    """Прогнать кейсы через Router API, собрать decisions."""
    results = []
    client = httpx.Client(base_url=url, timeout=timeout)

    for i, case in enumerate(cases):
        payload = {
            "query": case["query"],
            "risk_level": case.get("risk_level", "low"),
            "source_hint": case.get("source_hint", "none"),
            "require_json": case.get("require_json", False),
        }

        t0 = time.monotonic()
        try:
            r = client.post("/v1/route", json=payload,
                            headers={"X-Debug": "true", "Content-Type": "application/json"})
            r.raise_for_status()
            resp = r.json()
            latency = (time.monotonic() - t0) * 1000
        except Exception as e:
            resp = {"route": "error", "rag_mode": "none", "answer": str(e)}
            latency = (time.monotonic() - t0) * 1000

        results.append({
            "id": case["id"],
            "query": case["query"][:80],
            "category": case["category"],
            "expected_route": case["expected_route"],
            "actual_route": resp.get("route", "unknown"),
            "expected_rag_mode": case.get("expected_rag_mode", "none"),
            "actual_rag_mode": resp.get("rag_mode", "none"),
            "latency_ms": latency,
            "answer_len": len(resp.get("answer", "")),
            "confidence": resp.get("confidence", 0),
            "reason_codes": resp.get("reason_codes", []),
        })

        if (i + 1) % 10 == 0:
            console.print(f"  [{i+1}/{len(cases)}] processed")

    client.close()
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Вычислить все метрики."""
    total = len(results)
    if total == 0:
        return {}

    # Route accuracy
    correct = sum(1 for r in results if r["expected_route"] == r["actual_route"])
    route_accuracy = correct / total

    # Per-category accuracy
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["expected_route"] == r["actual_route"])
    cat_accuracy = {cat: sum(v) / len(v) for cat, v in by_cat.items()}

    # Confusion matrix
    confusion = defaultdict(Counter)
    for r in results:
        confusion[r["expected_route"]][r["actual_route"]] += 1

    # Secondary metrics
    # unnecessary_rag: actual has RAG, expected doesn't
    unnecessary_rag = sum(
        1 for r in results
        if r["expected_route"] == "direct_local"
        and r["actual_route"] in ("local_with_rag", "local_with_rag_and_repair")
    )
    unnecessary_rag_rate = unnecessary_rag / total

    # missed_rag: expected has RAG, actual doesn't
    rag_expected = [r for r in results if r["expected_route"] in ("local_with_rag", "local_with_rag_and_repair")]
    missed_rag = sum(1 for r in rag_expected if r["actual_route"] == "direct_local")
    missed_rag_rate = missed_rag / len(rag_expected) if rag_expected else 0

    # unnecessary_cloud: actual is cloud, expected isn't
    unnecessary_cloud = sum(
        1 for r in results
        if r["expected_route"] != "cloud_escalation"
        and r["actual_route"] == "cloud_escalation"
    )
    unnecessary_cloud_rate = unnecessary_cloud / total

    # missed_cloud: expected cloud, actual isn't
    cloud_expected = [r for r in results if r["expected_route"] == "cloud_escalation"]
    missed_cloud = sum(1 for r in cloud_expected if r["actual_route"] != "cloud_escalation")
    missed_cloud_rate = missed_cloud / len(cloud_expected) if cloud_expected else 0

    # Latency
    latencies = sorted(r["latency_ms"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return {
        "total": total,
        "route_accuracy": route_accuracy,
        "cat_accuracy": cat_accuracy,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "unnecessary_rag_rate": unnecessary_rag_rate,
        "missed_rag_rate": missed_rag_rate,
        "unnecessary_cloud_rate": unnecessary_cloud_rate,
        "missed_cloud_rate": missed_cloud_rate,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }


def print_report(metrics: dict):
    """Красивый отчёт."""
    console.print("\n[bold]═══ Routing Accuracy Report ═══[/bold]\n")

    # Main metrics
    table = Table(title="Primary Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status")

    acc = metrics["route_accuracy"]
    table.add_row("Route Accuracy", f"{acc:.1%}",
                  "≥ 75%", "[green]PASS" if acc >= 0.75 else "[red]FAIL")
    table.add_row("Unnecessary RAG", f"{metrics['unnecessary_rag_rate']:.1%}",
                  "< 20%", "[green]PASS" if metrics['unnecessary_rag_rate'] < 0.20 else "[red]FAIL")
    table.add_row("Missed RAG", f"{metrics['missed_rag_rate']:.1%}",
                  "< 15%", "[green]PASS" if metrics['missed_rag_rate'] < 0.15 else "[red]FAIL")
    table.add_row("Unnecessary Cloud", f"{metrics['unnecessary_cloud_rate']:.1%}",
                  "< 10%", "[green]PASS" if metrics['unnecessary_cloud_rate'] < 0.10 else "[red]FAIL")
    table.add_row("Missed Cloud", f"{metrics['missed_cloud_rate']:.1%}",
                  "< 10%", "[green]PASS" if metrics['missed_cloud_rate'] < 0.10 else "[red]FAIL")
    table.add_row("P50 Latency", f"{metrics['p50_latency_ms']:.0f}ms", "", "")
    table.add_row("P95 Latency", f"{metrics['p95_latency_ms']:.0f}ms", "", "")
    console.print(table)

    # Per-category
    table2 = Table(title="Per-Category Accuracy")
    table2.add_column("Category", style="cyan")
    table2.add_column("Accuracy", justify="right")
    for cat, acc in sorted(metrics["cat_accuracy"].items()):
        style = "[green]" if acc >= 0.75 else "[red]"
        table2.add_row(cat, f"{style}{acc:.1%}")
    console.print(table2)

    # Confusion matrix
    routes = sorted(set(
        list(metrics["confusion"].keys()) +
        [r for cm in metrics["confusion"].values() for r in cm]
    ))
    table3 = Table(title="Confusion Matrix (expected → actual)")
    table3.add_column("Expected \\ Actual", style="bold")
    for r in routes:
        table3.add_column(r[:15], justify="right")
    for exp in routes:
        row = [exp[:15]]
        for act in routes:
            val = metrics["confusion"].get(exp, {}).get(act, 0)
            style = "[green]" if exp == act and val > 0 else "[red]" if val > 0 else ""
            row.append(f"{style}{val}")
        table3.add_row(*row)
    console.print(table3)


def save_results(results: list[dict], metrics: dict, tag: str = "policy_c"):
    """Сохранить результаты."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    path = RESULTS_DIR / f"{tag}_{ts}.json"
    path.write_text(json.dumps({"results": results, "metrics": metrics}, indent=2, ensure_ascii=False))
    console.print(f"\nСохранено: {path}")


def main():
    parser = argparse.ArgumentParser(description="Routing accuracy analysis")
    parser.add_argument("--results", type=str, help="Load results from JSON file instead of running")
    parser.add_argument("--limit", type=int, help="Limit number of cases")
    parser.add_argument("--tag", type=str, default="policy_c", help="Tag for results file")
    parser.add_argument("--url", type=str, default=ROUTER_URL, help="Router API URL")
    args = parser.parse_args()

    golden = load_golden()
    if args.limit:
        golden = golden[:args.limit]

    console.print(f"[bold]Golden dataset: {len(golden)} cases[/bold]")

    if args.results:
        with open(args.results) as f:
            data = json.load(f)
            results = data["results"]
    else:
        console.print(f"Running through {args.url}...")
        results = run_through_router(golden, url=args.url)

    metrics = compute_metrics(results)
    print_report(metrics)
    save_results(results, metrics, tag=args.tag)


if __name__ == "__main__":
    main()
