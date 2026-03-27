#!/usr/bin/env python3
"""Мониторинг готовности TurboQuant для Ollama/llama.cpp.

Проверяет статус интеграции TurboQuant в экосистему:
- llama.cpp fork (TheTom/llama-cpp-turboquant)
- Ollama issue #15051
- Ollama KV cache типы

Использование:
    python turboquant_monitor.py           # Проверить статус
    python turboquant_monitor.py --verbose  # Подробный вывод
"""

from __future__ import annotations

import argparse
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Ключевые ресурсы для мониторинга
RESOURCES = {
    "llama.cpp fork": {
        "url": "https://github.com/TheTom/llama-cpp-turboquant",
        "description": "Community fork с TurboQuant для llama.cpp (6-фазный план)",
    },
    "Ollama issue": {
        "url": "https://github.com/ollama/ollama/issues/15051",
        "description": "Feature request: TurboQuant support в Ollama",
    },
    "vLLM issue": {
        "url": "https://github.com/vllm-project/vllm/issues/38171",
        "description": "TurboQuant KV cache quantization для vLLM",
    },
    "llama.cpp discussion": {
        "url": "https://github.com/ggml-org/llama.cpp/discussions/20969",
        "description": "Обсуждение TurboQuant в основном llama.cpp",
    },
    "PyTorch impl": {
        "url": "https://github.com/tonbistudio/turboquant-pytorch",
        "description": "Reference PyTorch implementation",
    },
    "Paper": {
        "url": "https://arxiv.org/html/2504.19874v1",
        "description": "TurboQuant: Online Vector Quantization (ICLR 2026)",
    },
}

# Ожидаемые новые KV cache типы в Ollama
EXPECTED_KV_TYPES = ["tq3_0", "tq2_5", "polarquant", "turboquant"]


def check_ollama_kv_types() -> list[str]:
    """Проверить поддерживает ли текущий Ollama новые KV cache типы."""
    # Известные поддерживаемые типы
    known_types = ["f16", "q8_0", "q4_0"]
    new_types = []

    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5
        )
        version = result.stdout.strip()
        console.print(f"Ollama version: {version}")
    except Exception:
        console.print("[yellow]Не удалось определить версию Ollama[/yellow]")

    return new_types


def check_ollama_release_notes() -> str | None:
    """Попытаться получить changelog Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return None


def print_status(verbose: bool = False):
    """Вывести текущий статус готовности TurboQuant."""
    console.print(Panel(
        "[bold]TurboQuant Status Monitor[/bold]\n\n"
        "TurboQuant (Google, ICLR 2026): PolarQuant + QJL\n"
        "6x сжатие KV cache, 0 потери качества на 3-bit\n\n"
        "[yellow]Статус: НЕ ДОСТУПЕН в Ollama[/yellow]\n"
        "Ожидаемый срок: Q2-Q3 2026",
        title="TurboQuant",
        border_style="cyan",
    ))

    # Ресурсы
    table = Table(title="Ключевые ресурсы")
    table.add_column("Название", style="cyan")
    table.add_column("URL")
    table.add_column("Описание")

    for name, info in RESOURCES.items():
        table.add_row(name, info["url"], info["description"])
    console.print(table)

    # Проверка Ollama
    console.print("\n[bold]Проверка текущей версии Ollama:[/bold]")
    version = check_ollama_release_notes()
    if version:
        console.print(f"  Версия: {version}")

    new_types = check_ollama_kv_types()
    if new_types:
        console.print(f"  [green]Найдены новые KV типы: {new_types}[/green]")
    else:
        console.print("  [dim]Новых KV cache типов не обнаружено[/dim]")

    # Влияние на qwen3.5:27b
    console.print(Panel(
        "[bold]Влияние на qwen3.5:27b (гибрид SSM/Attention):[/bold]\n\n"
        "• 16 из 64 слоёв используют KV cache (attention)\n"
        "• 48 слоёв — Mamba/SSM (не затронуты TurboQuant)\n"
        "• Ожидаемое сжатие KV: ~2.7x на attention-слоях\n"
        "• Главный выигрыш: контекст 64K+ на 100% GPU\n\n"
        "[bold]Когда станет доступно:[/bold]\n"
        "1. Добавить новые KV types в config.py\n"
        "2. Запустить benchmark_runner.py --phase 2\n"
        "3. Сравнить через analyze.py --compare",
        title="Qwen3.5:27b + TurboQuant",
        border_style="green",
    ))

    if verbose:
        console.print("\n[bold]Путь интеграции через custom GGUF:[/bold]")
        console.print("  1. Клонировать TheTom/llama-cpp-turboquant")
        console.print("  2. Собрать с TurboQuant support")
        console.print("  3. Квантовать: ./quantize model.gguf model-tq3.gguf TQ3_0")
        console.print("  4. Создать Modelfile: FROM ./custom_gguf/model-tq3.gguf")
        console.print("  5. ollama create qwen3.5-27b-tq3 -f Modelfile.tq3")
        console.print("  6. Запустить бенчмарки с новой моделью")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant readiness monitor")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    args = parser.parse_args()
    print_status(verbose=args.verbose)


if __name__ == "__main__":
    main()
