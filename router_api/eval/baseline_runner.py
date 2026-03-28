#!/usr/bin/env python3
"""Baseline runner: прогоняет golden dataset через разные политики. ANALYST-02.

Policy A: local only (llama-server напрямую, без роутера)
Policy B: local + RAG always-on (через роутер с source_hint=internal на всё)

Использование:
    python -m router_api.eval.baseline_runner --policy a --limit 20
    python -m router_api.eval.baseline_runner --policy b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

GOLDEN_PATH = Path(__file__).parent.parent / "golden_dataset.json"
LLAMA_URL = "http://127.0.0.1:11434"
ROUTER_URL = "http://127.0.0.1:8100"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "eval"


def load_golden(path: Path = GOLDEN_PATH, limit: int | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:limit] if limit else data


def run_policy_a(cases: list[dict]) -> list[dict]:
    """Policy A: прямой вызов llama-server, без роутера и RAG."""
    results = []
    client = httpx.Client(base_url=LLAMA_URL, timeout=120)

    for i, case in enumerate(cases):
        t0 = time.monotonic()
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "qwen3.5-27b",
                "messages": [{"role": "user", "content": case["query"]}],
                "temperature": 0.2,
                "max_tokens": 2048,
            })
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            timings = data.get("timings", {})
            tok_s = timings.get("predicted_per_second", 0)
            tokens = timings.get("predicted_n", 0)
        except Exception as e:
            content = ""
            tok_s = 0
            tokens = 0

        latency = (time.monotonic() - t0) * 1000
        results.append({
            "id": case["id"],
            "category": case["category"],
            "query": case["query"][:80],
            "policy": "a",
            "answer": content,
            "answer_len": len(content),
            "tokens": tokens,
            "tok_s": tok_s,
            "latency_ms": latency,
        })

        if (i + 1) % 10 == 0:
            console.print(f"  Policy A: [{i+1}/{len(cases)}]")

    client.close()
    return results


def run_policy_b(cases: list[dict]) -> list[dict]:
    """Policy B: через Router API с source_hint=internal на всё (RAG always-on)."""
    results = []
    client = httpx.Client(base_url=ROUTER_URL, timeout=120)

    for i, case in enumerate(cases):
        t0 = time.monotonic()
        try:
            r = client.post("/v1/route", json={
                "query": case["query"],
                "source_hint": "internal",  # Force RAG
                "require_json": case.get("require_json", False),
            }, headers={"Content-Type": "application/json", "X-Debug": "true"})
            r.raise_for_status()
            resp = r.json()
            content = resp.get("answer", "")
            route = resp.get("route", "unknown")
        except Exception as e:
            content = ""
            route = "error"

        latency = (time.monotonic() - t0) * 1000
        results.append({
            "id": case["id"],
            "category": case["category"],
            "query": case["query"][:80],
            "policy": "b",
            "route": route,
            "answer": content,
            "answer_len": len(content),
            "latency_ms": latency,
        })

        if (i + 1) % 10 == 0:
            console.print(f"  Policy B: [{i+1}/{len(cases)}]")

    client.close()
    return results


def compute_summary(results: list[dict], policy: str) -> dict:
    """Сводка по базовым метрикам."""
    latencies = sorted(r["latency_ms"] for r in results)
    has_answer = sum(1 for r in results if r["answer_len"] > 20)

    return {
        "policy": policy,
        "total": len(results),
        "has_answer": has_answer,
        "task_pass_rate": has_answer / len(results) if results else 0,
        "p50_latency": latencies[len(latencies) // 2] if latencies else 0,
        "p95_latency": latencies[int(len(latencies) * 0.95)] if latencies else 0,
        "avg_answer_len": sum(r["answer_len"] for r in results) / len(results) if results else 0,
    }


def print_summary(summary: dict):
    table = Table(title=f"Baseline: Policy {summary['policy'].upper()}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Cases", str(summary["total"]))
    table.add_row("Has Answer (>20 chars)", str(summary["has_answer"]))
    table.add_row("Task Pass Rate", f"{summary['task_pass_rate']:.1%}")
    table.add_row("P50 Latency", f"{summary['p50_latency']:.0f}ms")
    table.add_row("P95 Latency", f"{summary['p95_latency']:.0f}ms")
    table.add_row("Avg Answer Length", f"{summary['avg_answer_len']:.0f} chars")
    console.print(table)


def save_results(results: list[dict], summary: dict, policy: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"policy_{policy}_{ts}.json"
    path.write_text(json.dumps({"results": results, "summary": summary}, indent=2, ensure_ascii=False))
    console.print(f"Сохранено: {path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline runner")
    parser.add_argument("--policy", choices=["a", "b"], required=True, help="a=local-only, b=rag-always")
    parser.add_argument("--limit", type=int, help="Limit cases")
    args = parser.parse_args()

    golden = load_golden(limit=args.limit)
    console.print(f"[bold]Policy {args.policy.upper()}: {len(golden)} cases[/bold]")

    if args.policy == "a":
        results = run_policy_a(golden)
    else:
        results = run_policy_b(golden)

    summary = compute_summary(results, args.policy)
    print_summary(summary)
    save_results(results, summary, args.policy)


if __name__ == "__main__":
    main()
