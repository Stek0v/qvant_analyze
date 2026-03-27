#!/usr/bin/env bash
# switch-backend.sh — Переключение между Ollama и llama-server на порту 11434
#
# Использование:
#   sudo ./switch-backend.sh llama    # Переключить на llama-server
#   sudo ./switch-backend.sh ollama   # Переключить обратно на Ollama
#   ./switch-backend.sh status        # Показать текущий backend

set -euo pipefail

LLAMA_SERVICE="llama-server"
OLLAMA_SERVICE="ollama"
PORT=11434

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

status() {
    echo ""
    echo "═══ Статус backend-ов ═══"

    if systemctl is-active --quiet "$OLLAMA_SERVICE" 2>/dev/null; then
        echo -e "  Ollama:       ${GREEN}ACTIVE${NC}"
    else
        echo -e "  Ollama:       ${RED}stopped${NC}"
    fi

    if systemctl is-active --quiet "$LLAMA_SERVICE" 2>/dev/null; then
        echo -e "  llama-server: ${GREEN}ACTIVE${NC}"
    else
        echo -e "  llama-server: ${RED}stopped${NC}"
    fi

    # Кто слушает порт?
    local listener
    listener=$(ss -tlnp 2>/dev/null | grep ":${PORT}" | head -1 || true)
    if [[ -n "$listener" ]]; then
        echo -e "  Порт $PORT:    ${GREEN}занят${NC}"
        echo "    $listener"
    else
        echo -e "  Порт $PORT:    ${YELLOW}свободен${NC}"
    fi

    # GPU
    if command -v nvidia-smi &>/dev/null; then
        local vram
        vram=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
        echo "  GPU VRAM:     ${vram// /} MiB"
    fi

    # Проверка API
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo -e "  API health:   ${GREEN}OK (llama-server)${NC}"
    elif curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
        echo -e "  API health:   ${GREEN}OK (ollama)${NC}"
    else
        echo -e "  API health:   ${RED}недоступен${NC}"
    fi
    echo ""
}

wait_for_port_free() {
    echo "  Ожидание освобождения порта $PORT..."
    for i in $(seq 1 15); do
        if ! ss -tlnp 2>/dev/null | grep -q ":${PORT}"; then
            return 0
        fi
        sleep 1
    done
    echo -e "  ${RED}Порт $PORT не освободился за 15 секунд!${NC}"
    return 1
}

wait_for_api() {
    local endpoint="$1"
    echo "  Ожидание API..."
    for i in $(seq 1 30); do
        if curl -sf "$endpoint" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo -e "  ${RED}API не ответил за 30 секунд!${NC}"
    return 1
}

switch_to_llama() {
    echo ""
    echo "═══ Переключение на llama-server ═══"

    # Остановить Ollama
    if systemctl is-active --quiet "$OLLAMA_SERVICE" 2>/dev/null; then
        echo "  Останавливаю Ollama..."
        systemctl stop "$OLLAMA_SERVICE"
        systemctl disable "$OLLAMA_SERVICE" 2>/dev/null || true
    fi

    wait_for_port_free

    # Запустить llama-server
    echo "  Запускаю llama-server..."
    systemctl enable "$LLAMA_SERVICE" 2>/dev/null || true
    systemctl start "$LLAMA_SERVICE"

    wait_for_api "http://127.0.0.1:${PORT}/health"

    echo -e "  ${GREEN}✓ llama-server запущен на порту $PORT${NC}"
    echo ""
    echo "  API endpoints:"
    echo "    POST http://127.0.0.1:${PORT}/v1/chat/completions  (OpenAI-совместимый)"
    echo "    GET  http://127.0.0.1:${PORT}/health"
    echo ""
}

switch_to_ollama() {
    echo ""
    echo "═══ Переключение на Ollama ═══"

    # Остановить llama-server
    if systemctl is-active --quiet "$LLAMA_SERVICE" 2>/dev/null; then
        echo "  Останавливаю llama-server..."
        systemctl stop "$LLAMA_SERVICE"
        systemctl disable "$LLAMA_SERVICE" 2>/dev/null || true
    fi

    wait_for_port_free

    # Запустить Ollama
    echo "  Запускаю Ollama..."
    systemctl enable "$OLLAMA_SERVICE" 2>/dev/null || true
    systemctl start "$OLLAMA_SERVICE"

    wait_for_api "http://127.0.0.1:${PORT}/"

    echo -e "  ${GREEN}✓ Ollama запущен на порту $PORT${NC}"
    echo ""
    echo "  API endpoints:"
    echo "    POST http://127.0.0.1:${PORT}/api/chat      (Ollama API)"
    echo "    POST http://127.0.0.1:${PORT}/v1/chat/completions  (OpenAI-совместимый)"
    echo ""
}

# --- Main ---
case "${1:-status}" in
    llama|llama-server)
        if [[ $EUID -ne 0 ]]; then
            echo "Нужен sudo: sudo $0 llama"
            exit 1
        fi
        switch_to_llama
        status
        ;;
    ollama)
        if [[ $EUID -ne 0 ]]; then
            echo "Нужен sudo: sudo $0 ollama"
            exit 1
        fi
        switch_to_ollama
        status
        ;;
    status)
        status
        ;;
    *)
        echo "Использование: $0 {llama|ollama|status}"
        exit 1
        ;;
esac
