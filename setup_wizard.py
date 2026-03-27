#!/usr/bin/env python3
"""Интерактивный мастер настройки qvant для любой GPU + модели.

Использование:
    python setup_wizard.py                     # интерактивный
    python setup_wizard.py --auto              # автоматический
    python setup_wizard.py --model phi-4:14b   # конкретная модель
    python setup_wizard.py --gpu-override 8192 # переопределить VRAM
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gpu_profiles import (
    GPUSpec,
    ModelSpec,
    MODEL_DB,
    detect_gpu,
    get_model_spec,
    recommend_models,
    calculate_max_context,
    get_context_tiers,
)
from auto_config import (
    detect_hardware,
    generate_deployment,
    save_generated_files,
    save_hardware_profile,
    HardwareProfile,
    DeploymentConfig,
)

console = Console()
GENERATED_DIR = Path(__file__).parent / "generated"
RESULTS_DIR = Path(__file__).parent / "results"


# =====================================================================
# Phase A: Hardware Detection
# =====================================================================

def phase_a(gpu_override: int | None = None) -> HardwareProfile:
    console.print(Panel("[bold]Phase A: Определение железа[/bold]", border_style="blue"))

    hw = detect_hardware()

    if gpu_override:
        hw = HardwareProfile(
            gpu=GPUSpec(hw.gpu.name, gpu_override, hw.gpu.compute_cap,
                        hw.gpu.bandwidth_gbps, hw.gpu.supports_flash),
            cpu_cores=hw.cpu_cores, ram_mb=hw.ram_mb,
            disk_free_gb=hw.disk_free_gb, driver_version=hw.driver_version,
            llama_server_path=hw.llama_server_path,
        )
        console.print(f"  [yellow]GPU VRAM override: {gpu_override} MB[/yellow]")

    table = Table(title="Hardware Profile")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение")
    table.add_row("GPU", f"{hw.gpu.name}")
    table.add_row("VRAM", f"{hw.gpu.vram_mb} MB ({hw.gpu.vram_mb // 1024} GB)")
    table.add_row("Compute Capability", f"{hw.gpu.compute_cap}")
    table.add_row("Flash Attention", "да" if hw.gpu.supports_flash else "нет")
    table.add_row("CPU Cores", str(hw.cpu_cores))
    table.add_row("RAM", f"{hw.ram_mb} MB ({hw.ram_mb // 1024} GB)")
    table.add_row("Disk Free", f"{hw.disk_free_gb:.1f} GB")
    table.add_row("Driver", hw.driver_version)
    table.add_row("llama-server", hw.llama_server_path or "[red]не найден[/red]")
    console.print(table)

    # Сохраняем профиль
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_hardware_profile(hw, RESULTS_DIR / "hardware_profile.json")
    console.print(f"  Сохранено: results/hardware_profile.json\n")
    return hw


# =====================================================================
# Phase B: Model Selection
# =====================================================================

def phase_b(hw: HardwareProfile, model_name: str | None = None,
            auto: bool = False) -> ModelSpec:
    console.print(Panel("[bold]Phase B: Выбор модели[/bold]", border_style="blue"))

    recommendations = recommend_models(hw.gpu)

    if not recommendations:
        console.print("[red]Нет моделей, влезающих в GPU! Нужно больше VRAM.[/red]")
        raise SystemExit(1)

    # Если модель указана явно
    if model_name:
        try:
            model = get_model_spec(model_name)
            max_ctx = calculate_max_context(hw.gpu, model)
            if max_ctx < 2048:
                console.print(f"[red]{model_name} не влезает в {hw.gpu.vram_mb} MB VRAM![/red]")
                raise SystemExit(1)
            console.print(f"  Выбрана модель: [bold]{model.display_name}[/bold] ({model.quant}, {model.vram_mb} MB)")
            console.print(f"  Max context: {max_ctx}\n")
            return model
        except KeyError as e:
            console.print(f"[red]{e}[/red]")
            raise SystemExit(1)

    # Таблица рекомендаций
    table = Table(title=f"Модели для {hw.gpu.name} ({hw.gpu.vram_mb // 1024} GB)")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Модель", style="cyan")
    table.add_column("Quant")
    table.add_column("Размер", justify="right")
    table.add_column("Max Ctx", justify="right")
    table.add_column("Thinking")

    for i, (model, max_ctx) in enumerate(recommendations[:8], 1):
        ctx_label = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)
        table.add_row(
            str(i),
            model.display_name,
            model.quant,
            f"{model.vram_mb / 1024:.1f} GB",
            ctx_label,
            "да" if model.supports_thinking else "нет",
        )
    console.print(table)

    # Автоматический выбор — первая (самая большая)
    if auto:
        model, max_ctx = recommendations[0]
        console.print(f"  [green]Auto: {model.display_name}[/green]\n")
        return model

    # Интерактивный выбор
    choice = input(f"\n  Выберите модель [1-{min(8, len(recommendations))}]: ").strip()
    idx = int(choice) - 1
    if 0 <= idx < len(recommendations):
        model, _ = recommendations[idx]
        return model
    console.print("[red]Неверный выбор[/red]")
    raise SystemExit(1)


# =====================================================================
# Phase C: Configuration Generation
# =====================================================================

def phase_c(hw: HardwareProfile, model: ModelSpec) -> DeploymentConfig:
    console.print(Panel("[bold]Phase C: Генерация конфигурации[/bold]", border_style="blue"))

    # Путь к модели
    model_dir = Path("/usr/share/llama-server/models")
    model_path = str(model_dir / model.gguf_filename)

    config = generate_deployment(hw, model, model_path)

    # Показать конфигурацию
    table = Table(title="Параметры llama-server")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение")
    table.add_row("n-gpu-layers", str(config.n_gpu_layers))
    table.add_row("ctx-size", str(config.ctx_size))
    table.add_row("cache-type-k", config.cache_type_k)
    table.add_row("cache-type-v", config.cache_type_v)
    table.add_row("flash-attn", config.flash_attn)
    table.add_row("threads", str(config.threads))
    table.add_row("parallel", str(config.parallel))
    table.add_row("reasoning", config.reasoning)
    table.add_row("VRAM headroom", f"{config.vram_headroom_mb} MB")
    console.print(table)

    # Тиры контекста
    tiers = config.context_tiers
    console.print(f"  Context tiers:")
    console.print(f"    Core:     {tiers.get('core', [])}")
    console.print(f"    Extended: {tiers.get('extended', [])}")
    console.print(f"    Max:      {tiers.get('max', [])}")

    # Сохранить файлы
    files = save_generated_files(config, GENERATED_DIR)
    console.print(f"\n  Сгенерировано:")
    for name, path in files.items():
        console.print(f"    {name}: {path}")
    console.print()

    return config


# =====================================================================
# Phase D: Benchmark Plan
# =====================================================================

def phase_d(config: DeploymentConfig):
    console.print(Panel("[bold]Phase D: План бенчмарков[/bold]", border_style="blue"))

    tiers = config.context_tiers
    core = tiers.get("core", [])
    extended = tiers.get("extended", [])
    max_ctx = tiers.get("max", [])

    phase1_n = len(core) * 3  # 3 think modes
    phase2_n = len(core[-2:]) * 2 * 2 if len(core) >= 2 else 0  # 2 KV x 2 ctx x 2 think
    phase3_n = len(extended + max_ctx) * 4 if extended or max_ctx else 0  # 4 think modes
    phase4_n = 6  # temperature sweep

    prompts_p1 = 15  # 5 per set A/B/C
    prompts_p3 = 10  # Set C only
    prompts_p4 = 20  # Sets A+B

    table = Table(title="Оценка бенчмарков")
    table.add_column("Фаза", style="cyan")
    table.add_column("Конфигов", justify="right")
    table.add_column("Промптов", justify="right")
    table.add_column("Прогонов", justify="right")
    table.add_column("~Время", justify="right")

    total_runs = 0
    for label, n_cfg, n_prompt in [
        ("1 — Core", phase1_n, prompts_p1),
        ("2 — KV Cache", phase2_n, prompts_p1),
        ("3 — Extended", phase3_n, prompts_p3),
        ("4 — Temperature", phase4_n, prompts_p4),
    ]:
        runs = n_cfg * n_prompt
        total_runs += runs
        mins = runs * 40 // 60  # ~40 сек на прогон
        if runs > 0:
            table.add_row(label, str(n_cfg), str(n_prompt), str(runs), f"~{mins} мин")
        else:
            table.add_row(label, "—", "—", "—", "SKIP")

    table.add_row("[bold]Итого[/bold]", "", "", str(total_runs),
                  f"[bold]~{total_runs * 40 // 3600}ч {(total_runs * 40 % 3600) // 60}м[/bold]")
    console.print(table)

    console.print("\n[bold]Следующие шаги:[/bold]")
    if not Path(config.model_path).exists():
        console.print(f"  [yellow]1. Скачать модель:[/yellow]")
        console.print(f"     huggingface-cli download {config.model.hf_repo} --include '{config.model.gguf_filename}' --local-dir /usr/share/llama-server/models/")
    console.print(f"  2. Установить сервис:")
    console.print(f"     sudo cp generated/llama-server.env /etc/default/llama-server")
    console.print(f"     sudo cp generated/llama-server.service /etc/systemd/system/")
    console.print(f"     sudo systemctl daemon-reload && sudo systemctl restart llama-server")
    console.print(f"  3. Запустить бенчмарки:")
    console.print(f"     python benchmark_runner.py --phase 1")
    console.print(f"  4. Анализ:")
    console.print(f"     python analyze.py --all")
    console.print()


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="qvant setup wizard")
    parser.add_argument("--auto", action="store_true", help="Автоматический выбор")
    parser.add_argument("--model", type=str, default=None, help="Модель (напр. phi-4:14b)")
    parser.add_argument("--gpu-override", type=int, default=None, help="Override VRAM в MB")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]qvant Setup Wizard[/bold cyan]\n"
        "Автоконфигурация для любой NVIDIA GPU + модели",
        border_style="cyan",
    ))

    hw = phase_a(gpu_override=args.gpu_override)
    model = phase_b(hw, model_name=args.model, auto=args.auto)
    config = phase_c(hw, model)
    phase_d(config)

    console.print("[bold green]Setup wizard завершён.[/bold green]")


if __name__ == "__main__":
    main()
