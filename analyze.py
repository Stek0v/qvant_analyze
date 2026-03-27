#!/usr/bin/env python3
"""Анализ результатов бенчмарков qvant.

Использование:
    python analyze.py                    # Все результаты
    python analyze.py --phase 1          # Только фаза 1
    python analyze.py --export csv       # Экспорт в CSV
    python analyze.py --compare          # Сравнение конфигов
    python analyze.py --thinking-roi     # Анализ thinking ROI
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import RAW_DIR, REPORTS_DIR

console = Console()


def load_results(phase: int | None = None) -> list[dict]:
    """Загрузить все результаты из JSON файлов."""
    results = []
    for f in sorted(RAW_DIR.glob("*.json")):
        if phase and not f.name.startswith(f"phase{phase}_"):
            continue
        try:
            data = json.loads(f.read_text())
            results.append(data)
        except (json.JSONDecodeError, OSError):
            console.print(f"[yellow]Пропуск: {f.name}[/yellow]")
    return results


# ------------------------------------------------------------------
# Performance Matrix
# ------------------------------------------------------------------
def performance_matrix(results: list[dict]):
    """Таблица производительности по конфигурациям."""
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_config[r["config_name"]].append(r)

    table = Table(title="Performance Matrix")
    table.add_column("Конфиг", style="cyan", min_width=30)
    table.add_column("N", justify="right")
    table.add_column("Mean tok/s", justify="right", style="green")
    table.add_column("Median tok/s", justify="right")
    table.add_column("P95 tok/s", justify="right")
    table.add_column("Mean ms", justify="right")
    table.add_column("Mean Quality", justify="right", style="yellow")
    table.add_column("GPU%", style="bold")

    rows = []
    for name, group in sorted(by_config.items()):
        toks = [r["tokens_per_sec"] for r in group if r["tokens_per_sec"] > 0]
        durations = [r["total_duration_ms"] for r in group]
        quals = [r["quality_composite"] for r in group]

        if not toks:
            continue

        rows.append({
            "name": name,
            "n": len(group),
            "mean_toks": statistics.mean(toks),
            "median_toks": statistics.median(toks),
            "p95_toks": sorted(toks)[int(len(toks) * 0.05)] if len(toks) > 1 else toks[0],
            "mean_ms": statistics.mean(durations),
            "mean_qual": statistics.mean(quals),
            "gpu": group[-1].get("gpu_processor", "?"),
        })

    # Сортируем по mean tok/s
    rows.sort(key=lambda x: x["mean_toks"], reverse=True)

    for row in rows:
        table.add_row(
            row["name"],
            str(row["n"]),
            f"{row['mean_toks']:.1f}",
            f"{row['median_toks']:.1f}",
            f"{row['p95_toks']:.1f}",
            f"{row['mean_ms']:.0f}",
            f"{row['mean_qual']:.1f}",
            row["gpu"],
        )

    console.print(table)
    return rows


# ------------------------------------------------------------------
# Thinking ROI
# ------------------------------------------------------------------
def thinking_roi(results: list[dict]):
    """Анализ: стоит ли thinking того времени?"""
    # Группируем по (ctx, prompt_set, prompt_index)
    baseline: dict[tuple, list[dict]] = defaultdict(list)
    thinking_results: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r["num_ctx"], r["prompt_set"], r["prompt_index"])
        think_mode = r["think_mode"]
        if think_mode is False or think_mode == "false" or think_mode is None:
            baseline[key].append(r)
        else:
            thinking_results[key][str(think_mode)].append(r)

    table = Table(title="Thinking ROI Analysis")
    table.add_column("Context", justify="right")
    table.add_column("Think Mode", style="cyan")
    table.add_column("Pairs", justify="right")
    table.add_column("Avg Quality Δ", justify="right", style="green")
    table.add_column("Avg Time Δ (ms)", justify="right", style="red")
    table.add_column("ROI (Q/sec)", justify="right", style="yellow")
    table.add_column("Extra Tokens", justify="right")

    roi_data = []
    for key, base_list in baseline.items():
        ctx, pset, pidx = key
        base_q = statistics.mean([r["quality_composite"] for r in base_list])
        base_t = statistics.mean([r["total_duration_ms"] for r in base_list])

        for think_mode, think_list in thinking_results.get(key, {}).items():
            think_q = statistics.mean([r["quality_composite"] for r in think_list])
            think_t = statistics.mean([r["total_duration_ms"] for r in think_list])
            extra_tokens = statistics.mean([r["thinking_token_estimate"] for r in think_list])

            q_delta = think_q - base_q
            t_delta = think_t - base_t
            roi = (q_delta / (t_delta / 1000)) if t_delta > 0 else 0

            roi_data.append({
                "ctx": ctx, "think": think_mode,
                "q_delta": q_delta, "t_delta": t_delta,
                "roi": roi, "extra_tokens": extra_tokens,
            })

    # Агрегируем по (ctx, think_mode)
    agg: dict[tuple, list] = defaultdict(list)
    for d in roi_data:
        agg[(d["ctx"], d["think"])].append(d)

    for (ctx, think), group in sorted(agg.items()):
        avg_q = statistics.mean([d["q_delta"] for d in group])
        avg_t = statistics.mean([d["t_delta"] for d in group])
        avg_roi = statistics.mean([d["roi"] for d in group])
        avg_tok = statistics.mean([d["extra_tokens"] for d in group])

        q_style = "[green]" if avg_q > 0 else "[red]"
        table.add_row(
            str(ctx),
            think,
            str(len(group)),
            f"{q_style}{avg_q:+.1f}[/]",
            f"{avg_t:+.0f}",
            f"{avg_roi:.2f}",
            f"{avg_tok:.0f}",
        )

    console.print(table)


# ------------------------------------------------------------------
# VRAM Pressure
# ------------------------------------------------------------------
def vram_pressure(results: list[dict]):
    """Анализ давления на VRAM по контексту."""
    table = Table(title="VRAM Pressure by Context")
    table.add_column("Context", justify="right")
    table.add_column("KV Cache")
    table.add_column("Peak VRAM (MB)", justify="right")
    table.add_column("Headroom (MB)", justify="right")
    table.add_column("GPU%", style="bold")
    table.add_column("Spill?", style="red")

    by_ctx: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        by_ctx[(r["num_ctx"], r["kv_cache_type"])].append(r)

    for (ctx, kv), group in sorted(by_ctx.items()):
        peak = max(r["gpu_vram_peak_mb"] for r in group if r["gpu_vram_peak_mb"] > 0)
        from config import GPU_TOTAL_VRAM_MB
        headroom = GPU_TOTAL_VRAM_MB - peak if peak > 0 else 0
        gpu_pct = group[-1].get("gpu_processor", "?")
        spill = "YES" if "CPU" in gpu_pct else "no"

        headroom_style = "[green]" if headroom > 2000 else "[yellow]" if headroom > 500 else "[red]"

        table.add_row(
            str(ctx),
            kv,
            str(peak),
            f"{headroom_style}{headroom}[/]",
            gpu_pct,
            spill,
        )

    console.print(table)


# ------------------------------------------------------------------
# Optimal Profiles
# ------------------------------------------------------------------
def optimal_profiles(results: list[dict]):
    """Рекомендация оптимальных профилей fast/balanced/deep."""
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_config[r["config_name"]].append(r)

    candidates = []
    for name, group in by_config.items():
        toks = [r["tokens_per_sec"] for r in group if r["tokens_per_sec"] > 0]
        quals = [r["quality_composite"] for r in group]
        gpu_pct = group[-1].get("gpu_processor", "")
        is_gpu = "100% GPU" in gpu_pct

        if not toks:
            continue

        candidates.append({
            "name": name,
            "mean_toks": statistics.mean(toks),
            "mean_qual": statistics.mean(quals),
            "is_gpu": is_gpu,
            "config": group[0],
        })

    # Fast: максимум tok/s при 100% GPU
    gpu_only = [c for c in candidates if c["is_gpu"]]
    if not gpu_only:
        gpu_only = candidates

    fast = max(gpu_only, key=lambda c: c["mean_toks"], default=None)

    # Balanced: лучший баланс quality * tok/s при 100% GPU
    balanced = max(gpu_only, key=lambda c: c["mean_qual"] * c["mean_toks"], default=None)

    # Deep: максимум quality при 100% GPU
    deep = max(gpu_only, key=lambda c: c["mean_qual"], default=None)

    console.print("\n[bold]═══ Рекомендуемые профили ═══[/bold]")

    table = Table()
    table.add_column("Профиль", style="bold")
    table.add_column("Конфиг", style="cyan")
    table.add_column("Avg tok/s", justify="right", style="green")
    table.add_column("Avg Quality", justify="right", style="yellow")

    for label, choice in [("Fast", fast), ("Balanced", balanced), ("Deep", deep)]:
        if choice:
            table.add_row(
                label,
                choice["name"],
                f"{choice['mean_toks']:.1f}",
                f"{choice['mean_qual']:.1f}",
            )

    console.print(table)


# ------------------------------------------------------------------
# CSV Export
# ------------------------------------------------------------------
def export_csv(results: list[dict], output: Path | None = None):
    """Экспорт всех результатов в CSV."""
    if not results:
        console.print("[yellow]Нет данных для экспорта[/yellow]")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = output or REPORTS_DIR / "results.csv"

    fields = [
        "config_name", "prompt_set", "prompt_index", "model",
        "kv_cache_type", "num_ctx", "think_mode", "temperature",
        "tokens_per_sec", "prompt_tokens_per_sec",
        "total_duration_ms", "load_duration_ms",
        "prompt_eval_count", "prompt_eval_duration_ms",
        "eval_count", "eval_duration_ms",
        "is_cold_start", "gpu_vram_peak_mb", "gpu_processor",
        "quality_completeness", "quality_structure",
        "quality_conciseness", "quality_composite",
        "thinking_token_estimate", "content_token_estimate",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    console.print(f"[green]CSV экспортирован: {path}[/green]")


# ------------------------------------------------------------------
# Markdown Report
# ------------------------------------------------------------------
def generate_report(results: list[dict], phase: int | None = None):
    """Генерация markdown отчёта."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_phase{phase}" if phase else ""
    path = REPORTS_DIR / f"report{suffix}.md"

    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_config[r["config_name"]].append(r)

    lines = [f"# Отчёт бенчмарков qvant{f' (Фаза {phase})' if phase else ''}\n"]
    lines.append(f"Всего прогонов: {len(results)}\n")
    lines.append(f"Конфигураций: {len(by_config)}\n\n")

    lines.append("## Performance Matrix\n\n")
    lines.append("| Конфиг | N | Mean tok/s | Mean ms | Mean Quality | GPU% |\n")
    lines.append("|--------|---|-----------|---------|-------------|------|\n")

    for name, group in sorted(by_config.items()):
        toks = [r["tokens_per_sec"] for r in group if r["tokens_per_sec"] > 0]
        if not toks:
            continue
        durations = [r["total_duration_ms"] for r in group]
        quals = [r["quality_composite"] for r in group]
        gpu = group[-1].get("gpu_processor", "?")

        lines.append(
            f"| {name} | {len(group)} | {statistics.mean(toks):.1f} | "
            f"{statistics.mean(durations):.0f} | {statistics.mean(quals):.1f} | {gpu} |\n"
        )

    lines.append("\n---\nСгенерировано qvant analyze.py\n")

    path.write_text("".join(lines))
    console.print(f"[green]Отчёт: {path}[/green]")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="qvant benchmark analyzer")
    parser.add_argument("--phase", type=int, default=None, help="Фильтр по фазе")
    parser.add_argument("--export", choices=["csv"], help="Экспорт данных")
    parser.add_argument("--compare", action="store_true", help="Performance matrix")
    parser.add_argument("--thinking-roi", action="store_true", help="Thinking ROI analysis")
    parser.add_argument("--vram", action="store_true", help="VRAM pressure analysis")
    parser.add_argument("--profiles", action="store_true", help="Optimal profile recommendations")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--all", action="store_true", help="Все анализы")
    args = parser.parse_args()

    results = load_results(args.phase)
    if not results:
        console.print("[red]Нет результатов. Сначала запустите benchmark_runner.py[/red]")
        return

    console.print(f"[bold]Загружено результатов: {len(results)}[/bold]\n")

    run_all = args.all or not any([args.compare, args.thinking_roi, args.vram,
                                    args.profiles, args.export, args.report])

    if run_all or args.compare:
        performance_matrix(results)

    if run_all or args.thinking_roi:
        thinking_roi(results)

    if run_all or args.vram:
        vram_pressure(results)

    if run_all or args.profiles:
        optimal_profiles(results)

    if run_all or args.report:
        generate_report(results, args.phase)

    if args.export == "csv":
        export_csv(results)


if __name__ == "__main__":
    main()
