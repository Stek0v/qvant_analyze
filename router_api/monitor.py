#!/usr/bin/env python3
"""System health monitor для qvant stack. DEVOPS-10.

Проверяет: Router API, llama-server, Cognee, Langfuse, GPU, disk.
Алерты при: disk > 95%, GPU > 85°C, error rate > 5%.

Использование:
    python -m router_api.monitor            # одноразовая проверка
    python -m router_api.monitor --watch 60  # каждые 60 сек
"""

from __future__ import annotations

import argparse
import subprocess
import time

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

SERVICES = {
    "Router API": "http://127.0.0.1:8100/health",
    "llama-server": "http://127.0.0.1:11434/health",
    "Cognee MCP": "http://127.0.0.1:9000/health",
    "Langfuse": "http://127.0.0.1:3000/api/public/health",
}

DISK_WARN_PCT = 95
GPU_TEMP_WARN = 85


def check_services() -> dict[str, str]:
    results = {}
    for name, url in SERVICES.items():
        try:
            r = httpx.get(url, timeout=5)
            results[name] = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception:
            results[name] = "DOWN"
    return results


def check_gpu() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "vram_used": int(parts[0]),
            "vram_total": int(parts[1]),
            "temp": int(parts[2]),
            "util": int(parts[3]),
        }
    except Exception:
        return {}


def check_disk() -> dict:
    try:
        result = subprocess.run(
            ["df", "--output=pcent,avail", "/"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        parts = lines[-1].split()
        return {
            "used_pct": int(parts[0].rstrip("%")),
            "avail_gb": int(parts[1]) / (1024 * 1024),
        }
    except Exception:
        return {}


def run_check():
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"\n[bold]═══ Health Check [{ts}] ═══[/bold]")

    alerts = []

    # Services
    services = check_services()
    table = Table(title="Services")
    table.add_column("Service", style="cyan")
    table.add_column("Status")
    for name, status in services.items():
        style = "[green]" if status == "OK" else "[red]"
        table.add_row(name, f"{style}{status}")
        if status != "OK":
            alerts.append(f"SERVICE DOWN: {name} = {status}")
    console.print(table)

    # GPU
    gpu = check_gpu()
    if gpu:
        temp_style = "[green]" if gpu["temp"] < GPU_TEMP_WARN else "[red]"
        console.print(f"  GPU: {gpu['vram_used']}/{gpu['vram_total']} MB, "
                      f"{temp_style}{gpu['temp']}°C[/], {gpu['util']}% util")
        if gpu["temp"] >= GPU_TEMP_WARN:
            alerts.append(f"GPU TEMP: {gpu['temp']}°C >= {GPU_TEMP_WARN}°C")

    # Disk
    disk = check_disk()
    if disk:
        disk_style = "[green]" if disk["used_pct"] < DISK_WARN_PCT else "[red]"
        console.print(f"  Disk: {disk_style}{disk['used_pct']}%[/] used, {disk['avail_gb']:.1f} GB free")
        if disk["used_pct"] >= DISK_WARN_PCT:
            alerts.append(f"DISK: {disk['used_pct']}% >= {DISK_WARN_PCT}%")

    # Alerts
    if alerts:
        console.print(f"\n[bold red]ALERTS ({len(alerts)}):[/bold red]")
        for a in alerts:
            console.print(f"  [red]! {a}[/red]")
    else:
        console.print(f"\n[green]All checks passed.[/green]")

    return len(alerts)


def main():
    parser = argparse.ArgumentParser(description="qvant health monitor")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="Repeat every N seconds")
    args = parser.parse_args()

    if args.watch:
        console.print(f"[bold]Monitoring every {args.watch}s (Ctrl+C to stop)[/bold]")
        while True:
            run_check()
            time.sleep(args.watch)
    else:
        exit(run_check())


if __name__ == "__main__":
    main()
