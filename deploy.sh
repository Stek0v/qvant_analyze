#!/usr/bin/env bash
# deploy.sh — Развёртывание qvant на любой машине с NVIDIA GPU
#
# Использование:
#   ./deploy.sh                    # интерактивный
#   ./deploy.sh --auto             # автоматический
#   ./deploy.sh --model llama3.1:8b --auto
#
# Требования: NVIDIA GPU, Python 3.10+, nvidia-smi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

WIZARD_ARGS=("$@")

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║      qvant Deploy                     ║"
echo "║      Universal GPU + Model Optimizer  ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# --- Step 1: Prerequisites ---
echo -e "${GREEN}[1/5]${NC} Проверка зависимостей..."

# nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
    echo -e "${RED}nvidia-smi не найден! Установите NVIDIA драйверы.${NC}"
    exit 1
fi
echo "  ✓ nvidia-smi"

# Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}python3 не найден!${NC}"
    exit 1
fi
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Python ${PYVER}"

# pip packages
MISSING=""
for pkg in httpx rich tabulate tqdm numpy; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done
if [[ -n "$MISSING" ]]; then
    echo -e "${YELLOW}  Устанавливаю:${MISSING}${NC}"
    pip install $MISSING 2>&1 | tail -1
fi
echo "  ✓ Python packages"

# --- Step 2: llama-server ---
echo ""
echo -e "${GREEN}[2/5]${NC} Проверка llama-server..."

if command -v llama-server &>/dev/null; then
    echo "  ✓ llama-server: $(which llama-server)"
else
    echo -e "${YELLOW}  llama-server не найден.${NC}"
    echo "  Для установки:"
    echo "    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama-build"
    echo "    cd /tmp/llama-build && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
    echo "    cmake --build build --target llama-server -j\$(nproc)"
    echo "    sudo cp build/bin/llama-server /usr/local/bin/"
    echo "    sudo cp build/bin/lib*.so* /usr/local/lib/ && sudo ldconfig"
    echo ""
    read -p "  Установить автоматически? (y/n): " install_llama
    if [[ "${install_llama}" == "y" ]]; then
        echo "  Сборка llama-server (может занять 5-10 минут)..."
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama-build 2>&1 | tail -1
        cd /tmp/llama-build
        cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=native 2>&1 | tail -3
        cmake --build build --target llama-server -j$(nproc) 2>&1 | tail -3
        sudo cp build/bin/llama-server /usr/local/bin/
        sudo cp build/bin/lib*.so* /usr/local/lib/ 2>/dev/null || true
        sudo ldconfig 2>/dev/null || true
        cd "$SCRIPT_DIR"
        rm -rf /tmp/llama-build
        echo "  ✓ llama-server установлен"
    else
        echo -e "${RED}  Без llama-server продолжение невозможно.${NC}"
        exit 1
    fi
fi

# --- Step 3: Setup Wizard ---
echo ""
echo -e "${GREEN}[3/5]${NC} Запуск setup wizard..."
echo ""

python3 setup_wizard.py "${WIZARD_ARGS[@]}"

# --- Step 4: Install service ---
echo ""
echo -e "${GREEN}[4/5]${NC} Установка systemd сервиса..."

if [[ -f generated/llama-server.env ]] && [[ -f generated/llama-server.service ]]; then
    if [[ $EUID -eq 0 ]]; then
        cp generated/llama-server.env /etc/default/llama-server
        cp generated/llama-server.service /etc/systemd/system/llama-server.service
        systemctl daemon-reload
        echo "  ✓ Сервис установлен"
    else
        echo -e "${YELLOW}  Нужен sudo для установки сервиса:${NC}"
        echo "    sudo cp generated/llama-server.env /etc/default/llama-server"
        echo "    sudo cp generated/llama-server.service /etc/systemd/system/"
        echo "    sudo systemctl daemon-reload"
    fi
else
    echo -e "${YELLOW}  Файлы сервиса не сгенерированы (wizard не завершился?)${NC}"
fi

# --- Step 5: Summary ---
echo ""
echo -e "${GREEN}[5/5]${NC} Готово!"
echo ""
echo "  Следующие шаги:"
echo "    sudo systemctl start llama-server    # запустить сервер"
echo "    python3 benchmark_runner.py --phase 1  # бенчмарки"
echo "    python3 analyze.py --all             # анализ"
echo ""
