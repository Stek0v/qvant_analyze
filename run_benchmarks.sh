#!/usr/bin/env bash
# run_benchmarks.sh — Shell-обёртка для запуска фаз бенчмарков qvant
#
# Использование:
#   ./run_benchmarks.sh --phase 1              # Core matrix (без рестартов)
#   ./run_benchmarks.sh --phase 2              # KV cache comparison (с рестартами)
#   ./run_benchmarks.sh --phase 1 --dry-run    # Пробный запуск
#   ./run_benchmarks.sh --phase 1 --limit 3    # 3 промпта на конфиг

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHASE=""
DRY_RUN=""
LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)   PHASE="$2"; shift 2;;
        --dry-run) DRY_RUN="--dry-run"; shift;;
        --limit)   LIMIT="--limit $2"; shift 2;;
        *)         echo "Unknown: $1"; exit 1;;
    esac
done

if [[ -z "$PHASE" ]]; then
    echo "Использование: $0 --phase <1|2|3|4> [--dry-run] [--limit N]"
    exit 1
fi

# --- Проверка Ollama ---
wait_for_ollama() {
    echo "Ожидание Ollama..."
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:11434/ > /dev/null 2>&1; then
            echo "Ollama готов."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Ollama не отвечает!"
    return 1
}

# --- Рестарт Ollama с новым KV cache ---
restart_with_kv() {
    local kv_type="$1"
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  Рестарт Ollama с KV cache: $kv_type"
    echo "═══════════════════════════════════════════"

    # Создаём временный override
    local override_dir="/etc/systemd/system/ollama.service.d"
    local override_file="$override_dir/override.conf"

    # Читаем текущий override и меняем KV cache type
    if [[ -f "$override_file" ]]; then
        sudo sed -i "s/OLLAMA_KV_CACHE_TYPE=[a-z0-9_]*/OLLAMA_KV_CACHE_TYPE=$kv_type/" "$override_file"
    else
        echo "ERROR: override.conf не найден!"
        return 1
    fi

    sudo systemctl daemon-reload
    sudo systemctl restart ollama

    wait_for_ollama
    sleep 3  # дать время на стабилизацию

    echo "KV cache type: $kv_type"
    ollama ps 2>/dev/null || true
}

# --- Восстановление KV cache ---
restore_kv() {
    local original_kv="q8_0"
    echo ""
    echo "Восстановление KV cache: $original_kv"
    restart_with_kv "$original_kv"
}

# --- Запуск фазы ---
echo ""
echo "╔════════════════════════════════════╗"
echo "║     qvant Benchmark Runner         ║"
echo "║     Фаза: $PHASE                        ║"
echo "╚════════════════════════════════════╝"
echo ""

# Проверка Ollama
wait_for_ollama

if [[ "$PHASE" == "2" && -z "$DRY_RUN" ]]; then
    echo "Фаза 2 требует смены KV cache (f16, q4_0)."
    echo "Будет изменён /etc/systemd/system/ollama.service.d/override.conf"
    echo ""
    read -p "Продолжить? (y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "Отмена."
        exit 0
    fi

    # Запуск с f16
    restart_with_kv "f16"
    echo "--- Запуск бенчмарков с KV=f16 ---"
    # Для фазы 2 нужно запускать по частям
    # benchmark_runner сам спросит про KV cache

    # Запуск с q4_0
    restart_with_kv "q4_0"
    echo "--- Запуск бенчмарков с KV=q4_0 ---"

    # Восстановление
    restore_kv
fi

# Основной запуск
python3 benchmark_runner.py --phase "$PHASE" $DRY_RUN $LIMIT

echo ""
echo "═══ Готово ═══"
echo ""
echo "Следующие шаги:"
echo "  python3 analyze.py --phase $PHASE     # Анализ"
echo "  python3 analyze.py --export csv        # CSV экспорт"
echo "  python3 analyze.py --all               # Полный отчёт"
