#!/usr/bin/env python3
"""A/B comparison of routing policies. ANALYST-05.

Сравнивает все доступные eval результаты в единой таблице.

Использование:
    python -m router_api.eval.policy_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "eval"


def load_all_results() -> dict[str, dict]:
    """Загрузить все eval результаты, сгруппированные по policy tag."""
    policies = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        tag = f.stem  # e.g. "policy_a_20260328_061602"
        # Извлечь policy name
        if tag.startswith("policy_a"):
            name = "A: local only"
        elif tag.startswith("policy_b"):
            name = "B: local+RAG always"
        elif tag.startswith("policy_c_fixed"):
            name = "C: rules router (fixed)"
        elif tag.startswith("policy_c_levara"):
            name = "C: rules router"
        elif tag.startswith("policy_c"):
            name = "C: rules router (old)"
        elif tag.startswith("policy_d"):
            name = "D: semantic router"
        elif tag.startswith("smoke"):
            continue
        elif tag.startswith("nightly"):
            name = f"Nightly: {tag}"
        else:
            name = tag

        # Берём последний результат для каждой policy
        if "metrics" in data:
            policies[name] = data["metrics"]
        elif "summary" in data:
            # Baseline format
            s = data["summary"]
            policies[name] = {
                "total": s.get("total", 0),
                "task_pass_rate": s.get("task_pass_rate", 0),
                "p50_latency_ms": s.get("p50_latency", 0),
                "p95_latency_ms": s.get("p95_latency", 0),
                "route_accuracy": None,  # baselines don't have routing
            }

    return policies


def print_comparison():
    policies = load_all_results()

    if not policies:
        console.print("[red]No eval results found in results/eval/[/red]")
        return

    console.print(f"\n[bold]═══ Policy Comparison ({len(policies)} policies) ═══[/bold]\n")

    table = Table(title="Routing Metrics")
    table.add_column("Policy", style="cyan", min_width=25)
    table.add_column("Cases", justify="right")
    table.add_column("Route Acc", justify="right")
    table.add_column("Unnec. RAG", justify="right")
    table.add_column("Missed RAG", justify="right")
    table.add_column("Unnec. Cloud", justify="right")
    table.add_column("P50 ms", justify="right")
    table.add_column("P95 ms", justify="right")

    for name, m in sorted(policies.items()):
        acc = m.get("route_accuracy")
        acc_str = f"{acc:.1%}" if acc is not None else "n/a"
        ur = m.get("unnecessary_rag_rate")
        ur_str = f"{ur:.1%}" if ur is not None else "n/a"
        mr = m.get("missed_rag_rate")
        mr_str = f"{mr:.1%}" if mr is not None else "n/a"
        uc = m.get("unnecessary_cloud_rate")
        uc_str = f"{uc:.1%}" if uc is not None else "n/a"
        p50 = m.get("p50_latency_ms", m.get("p50_latency", 0))
        p95 = m.get("p95_latency_ms", m.get("p95_latency", 0))

        table.add_row(
            name,
            str(m.get("total", "?")),
            acc_str,
            ur_str,
            mr_str,
            uc_str,
            f"{p50:.0f}",
            f"{p95:.0f}",
        )

    console.print(table)

    # Winner
    routed = {n: m for n, m in policies.items() if m.get("route_accuracy") is not None}
    if routed:
        best = max(routed.items(), key=lambda x: x[1]["route_accuracy"])
        console.print(f"\n[bold green]Winner: {best[0]} ({best[1]['route_accuracy']:.1%} accuracy)[/bold green]")


def main():
    print_comparison()


if __name__ == "__main__":
    main()
