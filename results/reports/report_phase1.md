# Отчёт бенчмарков qvant (Фаза 1)
Всего прогонов: 137
Конфигураций: 9

## Performance Matrix

| Конфиг | N | Mean tok/s | Mean ms | Mean Quality | GPU% |
|--------|---|-----------|---------|-------------|------|
| q8kv_ctx16k_think-low_t02 | 16 | 33.5 | 50639 | 67.7 | 100% GPU |
| q8kv_ctx16k_think-medium_t02 | 16 | 33.9 | 49394 | 76.8 | 100% GPU |
| q8kv_ctx16k_think-off_t02 | 15 | 34.3 | 48294 | 71.8 | 100% GPU |
| q8kv_ctx4k_think-low_t02 | 15 | 33.9 | 48955 | 70.3 | 100% GPU |
| q8kv_ctx4k_think-medium_t02 | 15 | 34.9 | 47018 | 68.1 | 100% GPU |
| q8kv_ctx4k_think-off_t02 | 15 | 34.1 | 48647 | 72.7 | 100% GPU |
| q8kv_ctx8k_think-low_t02 | 15 | 34.7 | 49857 | 68.1 | 100% GPU |
| q8kv_ctx8k_think-medium_t02 | 15 | 34.5 | 50432 | 70.7 | 100% GPU |
| q8kv_ctx8k_think-off_t02 | 15 | 35.3 | 47348 | 67.4 | 100% GPU |

---
Сгенерировано qvant analyze.py
