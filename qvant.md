Ниже — уже оптимизированная инструкция под RTX 3090 24 GB + Ollama + qwen3.5:27b с сохранённой возможностью thinking, но без того, чтобы thinking по умолчанию убивал латентность и VRAM.

Главная идея такая:
сервер и модель настраиваются под стабильный быстрый режим, а thinking включается только на уровне запроса через поле think. В Ollama think можно передавать как true/false или как "low", "medium", "high" для поддерживаемых моделей, а thinking возвращается отдельно от обычного content.

Для qwen3.5:27b на одной 3090 базовым тегом должен быть именно 27b-q4_K_M: у него размер 17 GB, тогда как 27b-q8_0 — 30 GB, а 27b-bf16 — 56 GB. Это означает, что для одной 3090 разумный baseline — qwen3.5:27b/qwen3.5:27b-q4_K_M, а не q8_0 или bf16 как сам тег модели.

При этом KV cache в Ollama можно отдельно квантовать через OLLAMA_KV_CACHE_TYPE; доступные варианты — f16, q8_0, q4_0. Документация прямо рекомендует q8_0 как основной компромисс по памяти и качеству, а q4_0 — как более агрессивный режим с более заметной потерей точности на больших контекстах. Ollama также пишет, что квантованный KV cache работает при включённом Flash Attention.

1. Что именно надо изменить в прошлой инструкции

Оптимизация такая:

не включать thinking глобально
держать сервер в режиме Flash Attention + q8_0 KV cache
держать контекст умеренным
вызывать thinking только у отдельных запросов
разделить профили на fast / balanced / deep

Это лучше, чем одна “универсальная” конфигурация, потому что Ollama сам указывает, что память растёт вместе с OLLAMA_NUM_PARALLEL * OLLAMA_CONTEXT_LENGTH, а ollama ps показывает, загружена ли модель в 100% GPU или частично ушла в CPU.

2. Оптимальный server-level baseline

Открой override:

sudo systemctl edit ollama.service

Вставь:

[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_CONTEXT_LENGTH=16384"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_MAX_QUEUE=64"

Почему именно так:

OLLAMA_FLASH_ATTENTION=1 снижает расход памяти по мере роста контекста.
OLLAMA_KV_CACHE_TYPE=q8_0 уменьшает память KV примерно вдвое относительно f16 при обычно небольшой потере точности.
OLLAMA_NUM_PARALLEL=1 нужен потому, что Ollama прямо пишет: память масштабируется вместе с OLLAMA_NUM_PARALLEL * OLLAMA_CONTEXT_LENGTH.
OLLAMA_MAX_LOADED_MODELS=1 не даёт рантайму держать лишние модели на одной 3090.

Применение:

sudo systemctl daemon-reload
sudo systemctl restart ollama
3. Базовая модель

Подтяни модель:

ollama pull qwen3.5:27b

По текущим тегам это и есть рабочий 17 GB вариант q4_K_M.

Проверь модуль и размещение в памяти:

ollama ps

Тебе нужен режим:

100% GPU
CONTEXT: 16384 или то значение, которое ты выставил

Ollama прямо описывает, что 100% GPU значит полностью GPU-резидентную загрузку, а CPU/GPU — частичный оффлоад.

4. Правильная логика использования thinking
Не так

Не надо пытаться “встроить thinking” в system prompt, например фразами “думай глубже”, “рассуждай пошагово”. Это почти всегда даёт менее управляемый результат и мешает сравнивать режимы.

Так

Нужно управлять thinking только через API-поле think.

В Ollama /api/chat поддерживает:

think: false
think: true
think: "low"
think: "medium"
think: "high"
для поддерживаемых моделей, а thinking возвращается отдельным полем message.thinking.

Это и есть правильная оптимизация инструкции:
сервер быстрый всегда, thinking — только по запросу.

5. Рекомендуемая схема профилей
Профиль 1: Fast

Для 80–90% запросов.

{
  "model": "qwen3.5:27b",
  "think": false,
  "keep_alive": "30m",
  "options": {
    "num_ctx": 8192,
    "temperature": 0.2
  }
}

Использовать для:

обычного чата
RAG
суммаризации
извлечения фактов
кода без глубокого поиска багов
Профиль 2: Balanced Thinking

Для сложных, но не экстремальных задач.

{
  "model": "qwen3.5:27b",
  "think": "low",
  "keep_alive": "30m",
  "options": {
    "num_ctx": 12288,
    "temperature": 0.2
  }
}

Использовать для:

анализа архитектуры
рефакторинга
причинно-следственного разбора
сравнения 2–3 вариантов решения
Профиль 3: Deep Thinking

Только для дорогих reasoning-задач.

{
  "model": "qwen3.5:27b",
  "think": "medium",
  "keep_alive": "30m",
  "options": {
    "num_ctx": 16384,
    "temperature": 0.1
  }
}

Я бы не делал think: "high" дефолтным даже для сложных задач. На одной 3090 это уже режим “точечно и осознанно”, а не нормальный продовый baseline. Это уже инженерный вывод из общей стоимости генерации и размера модели.

6. Оптимизированный Modelfile

Лучше держать один нейтральный Modelfile, а thinking включать только в API.

Modelfile:

FROM qwen3.5:27b
PARAMETER num_ctx 16384
PARAMETER temperature 0.2
SYSTEM Ты отвечаешь ясно и по делу.
По умолчанию давай прямой ответ без лишних рассуждений.
Если запрос сложный, формируй структурированный ответ с выводом, рисками и рекомендациями.
Не повторяй вопрос пользователя.
Не растягивай ответ без необходимости.

PARAMETER num_ctx официально поддерживается в Modelfile.

Создание:

ollama create qwen3.5-27b-optimized -f Modelfile

Почему это лучше, чем агрессивный system prompt вроде “всегда думай глубоко”:
потому что модель не будет искусственно удлинять вывод даже там, где reasoning не нужен.

7. Готовые API-шаблоны
Быстрый режим
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5-27b-optimized",
  "messages": [
    {"role": "user", "content": "Сравни Docker Compose и Kubernetes для малого проекта"}
  ],
  "think": false,
  "stream": false,
  "keep_alive": "30m",
  "options": {
    "num_ctx": 8192,
    "temperature": 0.2
  }
}'
С thinking low
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5-27b-optimized",
  "messages": [
    {"role": "user", "content": "Сравни Docker Compose и Kubernetes для малого проекта, с рисками и компромиссами"}
  ],
  "think": "low",
  "stream": false,
  "keep_alive": "30m",
  "options": {
    "num_ctx": 12288,
    "temperature": 0.2
  }
}'
С thinking medium
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5-27b-optimized",
  "messages": [
    {"role": "user", "content": "Построй план миграции monolith -> microservices с этапами, рисками, метриками и критериями остановки"}
  ],
  "think": "medium",
  "stream": false,
  "keep_alive": "30m",
  "options": {
    "num_ctx": 16384,
    "temperature": 0.1
  }
}'
8. Как оптимизировать саму “инструкцию” для пользователя или приложения

Самая полезная форма инструкции для маршрутизации такая:

Режим ответа:
- Если вопрос простой, отвечай сразу и кратко.
- Если вопрос требует сравнения вариантов, включай структурированный анализ.
- Если вопрос требует многошагового вывода, допускается расширенное рассуждение.
- Не используй длинное рассуждение для коротких или справочных запросов.
- Сначала дай вывод, затем аргументацию.

Но ещё лучше — вообще не перекладывать маршрутизацию на модель, а делать это в приложении:

короткий factual prompt → think=false
архитектурный/аналитический prompt → think="low"
сложный planning/debugging → think="medium"

То есть оптимизированная инструкция — это не “заставить модель думать”, а правильно решать, когда thinking включать.

9. Оптимизированный план тестирования

Тестировать надо не только скорость, но и цену thinking.

В /api/chat Ollama возвращает:

total_duration
load_duration
prompt_eval_count
prompt_eval_duration
eval_count
eval_duration
и отдельное поле message.thinking, если thinking включён. Это достаточно для хорошей диагностики.
Матрица тестов

Минимум прогони такие режимы:

think=false, ctx=8192
think="low", ctx=12288
think="medium", ctx=16384
think=false, ctx=16384
think="low", ctx=16384
На каких задачах

Сделай 3 набора:

Набор A — короткие запросы

определение
краткое сравнение
извлечение фактов

Набор B — средняя сложность

выбор архитектуры
анализ trade-offs
разбор кода

Набор C — сложная логика

план миграции
сложный debugging
многошаговый reasoning
Что сравнивать

Смотри:

сохраняется ли 100% GPU
насколько растёт total_duration
сколько добавляет eval_count
реально ли улучшается качество ответа

Часто окажется, что:

для набора A thinking не нужен вообще,
для набора B low даёт лучший баланс,
для набора C medium оправдан, но уже дорогой.
10. Лучшая практическая конфигурация

Для твоего кейса я бы зафиксировал такой продовый baseline:

Сервер
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_CONTEXT_LENGTH=16384"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_MAX_QUEUE=64"
Модель

qwen3.5:27b / qwen3.5-27b-optimized

Режимы вызова
default: think=false
analysis: think="low"
deep analysis: think="medium"
Чего не делать
не ставить think=true глобально
не увеличивать num_ctx без теста ollama ps
не поднимать OLLAMA_NUM_PARALLEL выше 1, пока не доказано, что VRAM выдерживает
не использовать q4_0 KV cache как дефолт, пока не проверено качество на твоих задачах
11. Готовая “оптимизированная инструкция” в одном виде

Вот версия, которую можно положить в документацию проекта:

Базовый режим сервера:
- OLLAMA_FLASH_ATTENTION=1
- OLLAMA_KV_CACHE_TYPE=q8_0
- OLLAMA_CONTEXT_LENGTH=16384
- OLLAMA_NUM_PARALLEL=1
- OLLAMA_MAX_LOADED_MODELS=1
- OLLAMA_KEEP_ALIVE=30m

Базовая модель:
- qwen3.5:27b (17 GB q4_K_M tag)

Правила использования:
- По умолчанию использовать think=false.
- Для аналитических задач использовать think="low".
- Для самых сложных reasoning-задач использовать think="medium".
- Не использовать глобально think=true/high.
- Проверять ollama ps после каждого изменения контекста или KV cache.
- Цель — сохранить 100% GPU и минимизировать total_duration при приемлемом качестве.

Контексты:
- fast: 8192
- balanced: 12288
- deep: 16384

Проверки:
- ollama ps должен показывать 100% GPU
- сравнивать total_duration, prompt_eval_duration, eval_duration
- отдельно оценивать прирост качества от thinking



https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
