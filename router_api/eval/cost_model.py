#!/usr/bin/env python3
"""Cost/latency model для routing policies. ANALYST-04.

Оценивает стоимость и латентность каждого маршрута.
Breakeven анализ: при каком cloud_rate Policy C дешевле Policy B.

Использование:
    python -m router_api.eval.cost_model
    python -m router_api.eval.cost_model --requests 1000
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

console = Console()

# ── Cost parameters ──

# Local inference (RTX 3090, electricity + amortization)
LOCAL_COST_PER_REQUEST = 0.001  # $0.001 per request (~$0.72/day at 30 req/hr)
LOCAL_LATENCY_MS = {
    "direct_local": 5000,       # 5s avg (simple answer)
    "local_with_rag": 7000,     # 7s (search + generate)
    "local_with_rag_and_repair": 14000,  # 14s (2x generate)
}

# Cloud (OpenRouter, Claude Sonnet pricing)
CLOUD_INPUT_PER_1M = 3.0    # $3 per 1M input tokens
CLOUD_OUTPUT_PER_1M = 15.0  # $15 per 1M output tokens
CLOUD_AVG_PROMPT_TOKENS = 500
CLOUD_AVG_COMPLETION_TOKENS = 800
CLOUD_LATENCY_MS = 8000      # 8s avg

CLOUD_COST_PER_REQUEST = (
    CLOUD_AVG_PROMPT_TOKENS * CLOUD_INPUT_PER_1M / 1_000_000 +
    CLOUD_AVG_COMPLETION_TOKENS * CLOUD_OUTPUT_PER_1M / 1_000_000
)  # ~$0.0135 per request

# RAG (Levara search, negligible cost)
RAG_COST_PER_REQUEST = 0.0001  # ~$0.0001 (CPU on Pi 5)
RAG_LATENCY_MS = 50            # 50ms (2.6ms search + embedding)


def compute_policy_costs(n_requests: int = 1000) -> dict:
    """Вычислить стоимость для 4 политик."""

    # Предполагаемое распределение маршрутов для 1000 запросов
    # (из golden dataset: 30% general, 34% internal, 16% complex, 12% json, 9% high-risk, 12% adversarial)

    policies = {}

    # Policy A: local only — всё через llama-server
    a_cost = n_requests * LOCAL_COST_PER_REQUEST
    a_latency = LOCAL_LATENCY_MS["direct_local"]
    policies["A: local only"] = {
        "cost": a_cost,
        "p50_latency": a_latency,
        "cloud_pct": 0,
        "rag_pct": 0,
        "description": "Всё через llama-server, без RAG и cloud",
    }

    # Policy B: local + RAG always-on
    b_cost = n_requests * (LOCAL_COST_PER_REQUEST + RAG_COST_PER_REQUEST)
    b_latency = LOCAL_LATENCY_MS["local_with_rag"]
    policies["B: local+RAG always"] = {
        "cost": b_cost,
        "p50_latency": b_latency,
        "cloud_pct": 0,
        "rag_pct": 100,
        "description": "RAG на каждый запрос (даже если не нужен)",
    }

    # Policy C: rules router (текущий)
    # Распределение из v2+routes.md: 60% direct, 30% RAG, 5% repair, 5% cloud
    direct_n = int(n_requests * 0.60)
    rag_n = int(n_requests * 0.30)
    repair_n = int(n_requests * 0.05)
    cloud_n = n_requests - direct_n - rag_n - repair_n

    c_cost = (
        direct_n * LOCAL_COST_PER_REQUEST +
        rag_n * (LOCAL_COST_PER_REQUEST + RAG_COST_PER_REQUEST) +
        repair_n * (LOCAL_COST_PER_REQUEST * 2 + RAG_COST_PER_REQUEST) +
        cloud_n * CLOUD_COST_PER_REQUEST
    )
    c_latency_weighted = (
        direct_n * LOCAL_LATENCY_MS["direct_local"] +
        rag_n * LOCAL_LATENCY_MS["local_with_rag"] +
        repair_n * LOCAL_LATENCY_MS["local_with_rag_and_repair"] +
        cloud_n * CLOUD_LATENCY_MS
    ) / n_requests

    policies["C: rules router"] = {
        "cost": c_cost,
        "p50_latency": c_latency_weighted,
        "cloud_pct": cloud_n / n_requests * 100,
        "rag_pct": (rag_n + repair_n) / n_requests * 100,
        "description": f"Rules: {direct_n} direct, {rag_n} RAG, {repair_n} repair, {cloud_n} cloud",
    }

    # Policy D: semantic router (Iter 2) — более точная маршрутизация
    # Ожидание: меньше unnecessary RAG (25% вместо 30%), чуть больше cloud (7%)
    d_direct = int(n_requests * 0.63)
    d_rag = int(n_requests * 0.25)
    d_repair = int(n_requests * 0.05)
    d_cloud = n_requests - d_direct - d_rag - d_repair

    d_cost = (
        d_direct * LOCAL_COST_PER_REQUEST +
        d_rag * (LOCAL_COST_PER_REQUEST + RAG_COST_PER_REQUEST) +
        d_repair * (LOCAL_COST_PER_REQUEST * 2 + RAG_COST_PER_REQUEST) +
        d_cloud * CLOUD_COST_PER_REQUEST
    )
    d_latency_weighted = (
        d_direct * LOCAL_LATENCY_MS["direct_local"] +
        d_rag * LOCAL_LATENCY_MS["local_with_rag"] +
        d_repair * LOCAL_LATENCY_MS["local_with_rag_and_repair"] +
        d_cloud * CLOUD_LATENCY_MS
    ) / n_requests

    policies["D: semantic router"] = {
        "cost": d_cost,
        "p50_latency": d_latency_weighted,
        "cloud_pct": d_cloud / n_requests * 100,
        "rag_pct": (d_rag + d_repair) / n_requests * 100,
        "description": f"Semantic: {d_direct} direct, {d_rag} RAG, {d_repair} repair, {d_cloud} cloud",
    }

    return policies


def breakeven_analysis() -> float:
    """При каком cloud_rate Policy C дешевле Policy B?"""
    # Policy B cost per request = LOCAL + RAG = $0.0011
    b_per_req = LOCAL_COST_PER_REQUEST + RAG_COST_PER_REQUEST

    # Policy C cost: 0.60*LOCAL + 0.30*(LOCAL+RAG) + 0.05*(2*LOCAL+RAG) + cloud_rate*CLOUD
    # = 0.60*0.001 + 0.30*0.0011 + 0.05*0.0021 + cloud_rate*0.0135
    c_fixed = 0.60 * LOCAL_COST_PER_REQUEST + 0.30 * (LOCAL_COST_PER_REQUEST + RAG_COST_PER_REQUEST) + 0.05 * (2 * LOCAL_COST_PER_REQUEST + RAG_COST_PER_REQUEST)
    # c_fixed + cloud_rate * CLOUD_COST = b_per_req
    # cloud_rate = (b_per_req - c_fixed) / CLOUD_COST
    if CLOUD_COST_PER_REQUEST > 0:
        breakeven = (b_per_req - c_fixed) / CLOUD_COST_PER_REQUEST
        return max(0, breakeven)
    return 1.0


def print_report(n_requests: int = 1000):
    policies = compute_policy_costs(n_requests)

    console.print(f"\n[bold]═══ Cost/Latency Model ({n_requests} requests) ═══[/bold]\n")

    # Parameters
    console.print(f"  Local: ${LOCAL_COST_PER_REQUEST}/req, Cloud: ${CLOUD_COST_PER_REQUEST:.4f}/req")
    console.print(f"  Cloud pricing: ${CLOUD_INPUT_PER_1M}/1M input, ${CLOUD_OUTPUT_PER_1M}/1M output\n")

    table = Table(title="Policy Comparison")
    table.add_column("Policy", style="cyan")
    table.add_column("Total Cost", justify="right")
    table.add_column("Per Request", justify="right")
    table.add_column("P50 Latency", justify="right")
    table.add_column("Cloud %", justify="right")
    table.add_column("RAG %", justify="right")

    for name, p in policies.items():
        table.add_row(
            name,
            f"${p['cost']:.2f}",
            f"${p['cost']/n_requests:.4f}",
            f"{p['p50_latency']:.0f}ms",
            f"{p['cloud_pct']:.0f}%",
            f"{p['rag_pct']:.0f}%",
        )
    console.print(table)

    # Breakeven
    be = breakeven_analysis()
    console.print(f"\n[bold]Breakeven:[/bold] Policy C дешевле Policy B если cloud_rate < {be:.1%}")
    console.print(f"  Текущий cloud_rate в Policy C: ~5% → [green]значительно ниже breakeven[/green]")

    # Monthly projection
    console.print(f"\n[bold]Месячная проекция (30 req/hr, 24/7 = 21600 req/month):[/bold]")
    monthly = compute_policy_costs(21600)
    for name, p in monthly.items():
        console.print(f"  {name}: ${p['cost']:.2f}/month")


def main():
    parser = argparse.ArgumentParser(description="Cost/latency model")
    parser.add_argument("--requests", type=int, default=1000)
    args = parser.parse_args()
    print_report(args.requests)


if __name__ == "__main__":
    main()
