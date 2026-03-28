#!/usr/bin/env bash
# Nightly regression eval. DEVOPS-11.
# Cron: 0 2 * * * /home/stek0v/src/qvant/router_api/eval/nightly_regression.sh
#
# Прогоняет golden dataset, сравнивает с предыдущим, алерт при regression.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="$PROJECT_DIR/results/eval/nightly"
LOG="$RESULTS_DIR/$(date +%Y-%m-%d).log"
TAG="nightly_$(date +%Y%m%d)"

mkdir -p "$RESULTS_DIR"

echo "$(date): Nightly regression started" > "$LOG"

# Проверить что Router API и llama-server живы
if ! curl -sf http://127.0.0.1:8100/health > /dev/null 2>&1; then
    echo "$(date): ALERT — Router API unreachable!" >> "$LOG"
    exit 1
fi

# Прогнать golden dataset
cd "$PROJECT_DIR"
python3 -m router_api.eval.routing_accuracy --tag "$TAG" >> "$LOG" 2>&1

# Найти последний результат
LATEST=$(ls -t "$PROJECT_DIR/results/eval/${TAG}"*.json 2>/dev/null | head -1)
if [[ -z "$LATEST" ]]; then
    echo "$(date): ALERT — No results file generated!" >> "$LOG"
    exit 1
fi

# Извлечь метрики
ACCURACY=$(python3 -c "
import json
with open('$LATEST') as f:
    d = json.load(f)
m = d.get('metrics', {})
print(f\"{m.get('route_accuracy', 0):.3f}\")
")

echo "$(date): Route accuracy = $ACCURACY" >> "$LOG"

# Алерт если accuracy < 0.75
if python3 -c "exit(0 if float('$ACCURACY') >= 0.75 else 1)"; then
    echo "$(date): PASS — accuracy >= 75%" >> "$LOG"
else
    echo "$(date): ALERT — accuracy $ACCURACY < 75%!" >> "$LOG"
    # TODO: webhook/email alert
fi

echo "$(date): Nightly regression completed" >> "$LOG"
