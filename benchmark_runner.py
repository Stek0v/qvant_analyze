#!/usr/bin/env python3
"""Главный оркестратор бенчмарков qvant.

Использование:
    python benchmark_runner.py --phase 1
    python benchmark_runner.py --phase 1 --dry-run
    python benchmark_runner.py --phase 1 --limit 3   # только 3 промпта на конфиг
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from config import (
    RAW_DIR,
    COOLDOWN_SECONDS,
    BenchmarkResult,
    TestConfig,
    build_phase1_configs,
    build_phase2_configs,
    build_phase3_configs,
    build_phase4_configs,
)
from gpu_monitor import GPUMonitor, snapshot
from ollama_client import InferenceClient
from quality_scorer import score_response
from test_prompts import get_prompts_for_phase

console = Console()


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_result(result: BenchmarkResult, phase: int):
    """Сохранить результат в JSON."""
    path = RAW_DIR / f"phase{phase}_{result.config_name}_{result.prompt_set}{result.prompt_index}_{result.run_id[:8]}.json"
    data = dataclasses.asdict(result)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path


def validate_environment(client: InferenceClient) -> bool:
    """Проверить что inference server запущен и модель доступна."""
    console.print(f"[bold]Проверка окружения ({client.backend})...[/bold]")

    if not client.wait_ready():
        console.print("[red]Сервер не отвечает![/red]")
        return False
    console.print(f"  [green]✓[/green] {client.backend} доступен")

    gpu_pct = client.get_gpu_percent()
    console.print(f"  [green]✓[/green] GPU: {gpu_pct}")

    try:
        gpu = snapshot()
        console.print(f"  [green]✓[/green] GPU: {gpu.vram_used_mb}/{gpu.vram_total_mb} MB, {gpu.temp_c}°C")
    except Exception as e:
        console.print(f"  [red]✗[/red] nvidia-smi: {e}")
        return False

    return True


def run_single(
    client: InferenceClient,
    monitor: GPUMonitor,
    config: TestConfig,
    prompt_set: str,
    prompt_index: int,
    prompt_text: str,
    phase: int,
    dry_run: bool = False,
) -> BenchmarkResult | None:
    """Выполнить один бенчмарк-прогон."""

    if dry_run:
        console.print(f"  [dim]DRY RUN: {config.name} | {prompt_set}{prompt_index}[/dim]")
        return None

    # GPU snapshot до
    try:
        pre_gpu = snapshot()
    except Exception:
        pre_gpu = None

    # Инференс с мониторингом GPU
    def _do_inference():
        return client.chat(
            messages=[{"role": "user", "content": prompt_text}],
            model=config.model,
            think=config.think,
            num_ctx=config.num_ctx,
            temperature=config.temperature,
            top_p=config.top_p,
            presence_penalty=config.presence_penalty,
        )

    result, gpu_snapshots = monitor.monitor_during(_do_inference)

    # GPU metrics
    gpu_summary = GPUMonitor.summarize(gpu_snapshots)
    result.config_name = config.name
    result.kv_cache_type = config.kv_cache_type
    result.prompt_set = prompt_set
    result.prompt_index = prompt_index
    result.prompt_text = prompt_text
    result.gpu_vram_pre_mb = pre_gpu.vram_used_mb if pre_gpu else 0
    result.gpu_vram_peak_mb = gpu_summary["peak_vram_mb"]
    result.gpu_vram_post_mb = gpu_snapshots[-1].vram_used_mb if gpu_snapshots else 0
    result.gpu_util_avg_pct = gpu_summary["avg_util_pct"]
    result.gpu_temp_max_c = gpu_summary["max_temp_c"]
    result.gpu_processor = client.get_gpu_percent()

    # Quality scoring
    scores = score_response(result.content, prompt_set, prompt_index, result.thinking)
    result.quality_completeness = scores.completeness
    result.quality_structure = scores.structure
    result.quality_conciseness = scores.conciseness
    result.quality_composite = scores.composite

    return result


def run_config_group(
    client: InferenceClient,
    monitor: GPUMonitor,
    config: TestConfig,
    prompts: list[tuple[str, int, str]],
    phase: int,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """Прогнать все промпты для одного конфига."""
    results = []

    # Warmup: загрузить модель с нужным контекстом
    if not dry_run:
        console.print(f"\n[bold cyan]Warmup[/bold cyan]: {config.name} (ctx={config.num_ctx})")
        warmup = client.warmup(model=config.model, num_ctx=config.num_ctx)
        if warmup.is_cold_start:
            console.print(f"  [yellow]Cold start: load={warmup.load_duration_ms:.0f}ms[/yellow]")

        # Проверка GPU%
        gpu_pct = client.get_gpu_percent()
        if "100% GPU" not in gpu_pct:
            console.print(f"  [red]⚠ CPU spill: {gpu_pct}[/red]")
        else:
            console.print(f"  [green]✓ {gpu_pct}[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{config.name}", total=len(prompts))

        for set_name, idx, text in prompts:
            result = run_single(
                client, monitor, config, set_name, idx, text, phase, dry_run
            )
            if result:
                path = save_result(result, phase)
                results.append(result)

                # Краткий статус
                tok_s = f"{result.tokens_per_sec:.1f} tok/s"
                qual = f"Q={result.quality_composite:.0f}"
                progress.console.print(
                    f"    {set_name}{idx}: {tok_s} | {result.total_duration_ms:.0f}ms | {qual}"
                )

            if not dry_run:
                time.sleep(COOLDOWN_SECONDS)

            progress.advance(task)

    return results


def run_phase(phase: int, dry_run: bool = False, limit: int | None = None):
    """Запустить указанную фазу бенчмарков."""
    ensure_dirs()

    # Получить конфиги
    if phase == 1:
        configs = build_phase1_configs()
    elif phase == 2:
        configs = build_phase2_configs()
    elif phase == 3:
        configs = build_phase3_configs()
    elif phase == 4:
        configs = build_phase4_configs()
    else:
        console.print(f"[red]Неизвестная фаза: {phase}[/red]")
        return

    prompts = get_prompts_for_phase(phase)
    if limit:
        prompts = prompts[:limit]

    console.print(f"\n[bold]═══ Фаза {phase} ═══[/bold]")
    console.print(f"Конфигураций: {len(configs)}")
    console.print(f"Промптов: {len(prompts)}")
    console.print(f"Всего прогонов: {len(configs) * len(prompts)}")
    if dry_run:
        console.print("[yellow]DRY RUN — запросы не отправляются[/yellow]")
    console.print()

    # Группировка по KV cache type для минимизации перезапусков
    kv_groups: dict[str, list[TestConfig]] = {}
    for cfg in configs:
        kv_groups.setdefault(cfg.kv_cache_type, []).append(cfg)

    client = InferenceClient()
    monitor = GPUMonitor(interval=0.5)
    console.print(f"[bold]Backend: {client.backend}[/bold]")

    if not dry_run and not validate_environment(client):
        sys.exit(1)

    all_results: list[BenchmarkResult] = []

    for kv_type, group_configs in kv_groups.items():
        console.print(f"\n[bold magenta]KV Cache: {kv_type}[/bold magenta]")

        if not dry_run and kv_type != "q8_0":
            console.print(f"[yellow]⚠ Для KV cache '{kv_type}' нужен рестарт Ollama![/yellow]")
            console.print(f"  Запустите: sudo OLLAMA_KV_CACHE_TYPE={kv_type} systemctl restart ollama")
            console.print("  Или используйте run_benchmarks.sh --phase 2")
            resp = input("  Продолжить с текущим KV cache? (y/n): ").strip().lower()
            if resp != "y":
                console.print("[dim]Пропускаю группу...[/dim]")
                continue

        # Внутри KV группы: сортировка по ctx для минимума перезагрузок
        group_configs.sort(key=lambda c: c.num_ctx)

        for cfg in group_configs:
            results = run_config_group(client, monitor, cfg, prompts, phase, dry_run)
            all_results.extend(results)

    # Финальная сводка
    if all_results:
        print_summary(all_results, phase)

    client.close()
    console.print(f"\n[bold green]Фаза {phase} завершена. Результатов: {len(all_results)}[/bold green]")


def print_summary(results: list[BenchmarkResult], phase: int):
    """Краткая сводка по завершении фазы."""
    console.print(f"\n[bold]═══ Сводка фазы {phase} ═══[/bold]")

    # Группируем по конфигу
    by_config: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_config.setdefault(r.config_name, []).append(r)

    table = Table(title=f"Результаты фазы {phase}")
    table.add_column("Конфиг", style="cyan")
    table.add_column("Прогонов", justify="right")
    table.add_column("Avg tok/s", justify="right")
    table.add_column("Avg ms", justify="right")
    table.add_column("Avg Quality", justify="right")
    table.add_column("Peak VRAM", justify="right")
    table.add_column("GPU%", style="green")

    for name, group in by_config.items():
        avg_toks = sum(r.tokens_per_sec for r in group) / len(group)
        avg_ms = sum(r.total_duration_ms for r in group) / len(group)
        avg_q = sum(r.quality_composite for r in group) / len(group)
        peak_vram = max(r.gpu_vram_peak_mb for r in group)
        gpu_pct = group[-1].gpu_processor

        table.add_row(
            name,
            str(len(group)),
            f"{avg_toks:.1f}",
            f"{avg_ms:.0f}",
            f"{avg_q:.1f}",
            f"{peak_vram} MB",
            gpu_pct,
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="qvant benchmark runner")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Фаза тестирования (1-4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Пробный запуск без реальных запросов")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ограничить количество промптов на конфиг")
    parser.add_argument("--model", type=str, default=None,
                        help="Override модели (напр. llama3.1:8b)")
    parser.add_argument("--ctx", type=int, nargs="+", default=None,
                        help="Override контекстов (напр. --ctx 2048 4096 8192)")
    parser.add_argument("--backend-url", type=str, default=None,
                        help="Override URL сервера")
    args = parser.parse_args()

    # Override модели в config
    if args.model:
        import config as _cfg
        _cfg.MODEL_DEFAULT = args.model

    run_phase(args.phase, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
