Да. С учётом ограничения “не опираться на LiteLLM как на центральный gateway”, я бы строил стек так: свой тонкий router/control-plane, локальный llama-server как основной inference backend, твой модифицированный cognito как RAG-слой, и отдельный контур наблюдаемости и eval. Это даст меньше attack surface и лучшее понимание, где именно система ошибается: в маршрутизации, retrieval, генерации или эскалации.

1. Рекомендуемый стек
A. Ingress и изоляция

Снаружи — только reverse proxy и сеть. Сам router, llama-server, RAG и eval-инструменты лучше держать во внутреннем сегменте. Это не “AI-компонент”, а граница доверия: rate limit, auth, mTLS/VPN, allowlist клиентов, аудит запросов.

B. Router API

FastAPI как собственный gateway/router. Он подходит именно для такого сервиса: это современный high-performance API framework на Python с типизацией и встроенной автоматической документацией/OpenAPI, поэтому на нём удобно делать нормализацию запросов, policy checks, routing decisions и service-to-service API.

Что живёт в этом сервисе:

нормализация входа;
hard rules;
semantic routing;
решение “нужен ли RAG”;
решение “нужна ли эскалация в cloud”;
бюджетные лимиты;
журнал маршрутизации.
C. Локальный inference

llama-server из llama.cpp — основной исполнитель. У него есть OpenAI-compatible chat/responses/embeddings routes, Anthropic-compatible chat completions, continuous batching, monitoring endpoints и schema-constrained JSON response format. Это делает его удобным локальным inference-слоем под твой Qwen на 3090 без дополнительного gateway-посредника.

Роль:

default path для большинства запросов;
structured output;
first-pass reasoning;
first-pass repair;
локальный judge/verifier для дешёвой проверки перед cloud escalation.
D. RAG-слой

Твой модифицированный Cognito оставляй как основной knowledge layer. Его не надо ломать ради роутера. Router должен лишь решать:

идти без retrieval;
идти в shallow retrieval;
идти в deep retrieval;
делать повторный retrieval;
отправлять в cloud после локального RAG.
E. Semantic router

Для слоя “какой пайплайн нужен” я бы взял Semantic Router. Он позиционируется как “superfast decision-making layer” для LLM/agents и делает routing через semantic vector space вместо генерации отдельного LLM-ответа; у него есть интеграции с FastEmbed, Hugging Face и Qdrant.

Его роль у тебя:

intent classification;
pipeline selection;
дешёвое разделение на direct / rag / rag+verify / cloud-candidate.
F. Observability

Для трассировки и анализа — Langfuse self-hosted. Он open-source, self-hostable через Docker/VM/Kubernetes и даёт observability: trace logging, transparency, cost и latency analysis.

Роль:

лог всех маршрутов;
latency и cost;
сравнение политик роутинга;
просмотр неудачных кейсов;
накопление корпуса для последующего обучения роутера.
G. Offline eval / regression

Для пакетных тестов — Promptfoo. Это open-source CLI/library для evaluating и red-teaming LLM apps; есть self-hosting, web viewer и CI/CD-сценарии.

Для judge-based метрик — DeepEval. Он позиционируется как open-source evaluation framework, даёт 30+ метрик и G-Eval для custom LLM-evaluated scoring.

H. Опционально: retrieval backend upgrade

Если твой текущий retrieval в Cognito слаб на запросах с важными ключевыми словами или смешанном поиске, имеет смысл опционально подключить Qdrant как backend для hybrid search. Его документация описывает dense+sparse/hybrid queries и серверную Query API для улучшения retrieval quality.

2. Какой стек я бы выбрал
Вариант 1 — минимальный и самый безопасный

Подходит, если хочешь быстро запуститься и минимизировать сложность.

Состав:

reverse proxy;
свой Router API на FastAPI;
llama-server;
Cognito RAG;
Langfuse;
Promptfoo.

Логика:

hard rules;
rule-based decision “нужен ли RAG”;
local-first;
cloud только по ручным триггерам.

Плюсы:

минимальный риск;
проще отладка;
меньше moving parts.

Минусы:

хуже покрывает пограничные intent-case’ы;
больше ручной настройки правил.
Вариант 2 — мой основной выбор

Это лучший баланс.

Состав:

reverse proxy;
Router API на FastAPI;
Semantic Router;
llama-server;
Cognito RAG;
Langfuse;
Promptfoo + DeepEval.

Логика:

hard rules;
semantic pipeline routing;
retrieval routing;
local generation;
local verification;
cloud escalation.

Плюсы:

хорошая экономия;
меньше unnecessary RAG;
меньше unnecessary cloud;
остаётся контролируемым.

Минусы:

чуть сложнее развернуть;
нужно калибровать пороги уверенности.
Вариант 3 — продвинутый

Имеет смысл только после 2–4 недель логов.

Состав:

всё из варианта 2;
learned escalation scorer;
optional hybrid retrieval через Qdrant;
отдельный reranking layer.

Плюсы:

максимальная эффективность ресурсов;
лучше на сложных кейсах.

Минусы:

без накопленных данных почти не окупается.
3. Архитектура роутинга без LiteLLM

Я бы делал не один роутер, а каскад из трёх.

Router 1 — Pipeline Router

Решает:

direct_local
local_with_rag
local_with_rag_and_verify
cloud_candidate

Это либо hard rules, либо Semantic Router.

Router 2 — RAG Router

Решает:

no_rag
shallow_rag
deep_rag
requery_rag

Он должен смотреть не только на текст запроса, но и на retrieval signals:

score top-1;
разрыв между top-1 и top-2;
число релевантных документов;
наличие явного answer span;
keyword-heavy nature вопроса.
Router 3 — Escalation Router

Решает:

локальный ответ нормальный;
нужен repair;
нужен cloud.

Этот роутер я бы сначала делал правилами, не ML:

низкая уверенность;
конфликт с retrieval evidence;
не пройден JSON/schema check;
не пройден factuality/groundedness check;
high-risk task;
повторный локальный фейл.
4. Варианты использования
Сценарий 1 — обычный вопрос без внутреннего контекста

Путь:
router -> direct_local -> local answer

Это 60–80% дешёвых запросов.

Сценарий 2 — вопрос по внутренним данным

Путь:
router -> local_with_rag -> cognito -> local answer

Основной рабочий режим для твоего knowledge use-case.

Сценарий 3 — сложный вопрос по нескольким внутренним источникам

Путь:
router -> deep_rag -> synthesize -> local verify -> answer

Подходит для сравнений, расхождений, анализа нескольких документов.

Сценарий 4 — локальный ответ сомнительный

Путь:
local_with_rag -> local verify fail -> cloud escalation

Облако подключается только после признаков провала.

Сценарий 5 — high-risk / критичный формат / строгая точность

Путь:
hard rule -> cloud_candidate -> cloud

Таких кейсов должно быть мало и они должны быть формально описаны.

5. Полный состав компонентов

Ниже — именно тот набор, который я бы считал “полным”, но без избыточности.

Reverse proxy / access boundary
Router API на FastAPI
Policy engine внутри Router API
Semantic Router
RAG Router logic
Escalation Router logic
llama-server / llama.cpp
Cognito RAG
Cloud adapter к 1–2 облачным моделям
Langfuse
Promptfoo
DeepEval
Набор golden datasets
Хранилище логов и результатов eval
Скрипты nightly regression
Дашборд метрик и алертов

Опционально:
17. Qdrant для hybrid retrieval, если текущий retrieval не тянет keyword-heavy и mixed search кейсы.

6. Полный список для тестирования

Ниже — тот тест-план, который реально позволит выбрать оптимальный роутер или каскад роутеров.

I. Базовые функциональные тесты
Запрос проходит через direct_local.
Запрос проходит через local_with_rag.
Запрос уходит в deep_rag.
Запрос эскалируется в cloud.
Запрос с пустым history работает.
Запрос с длинным history работает.
Запрос с attachment metadata не ломает маршрут.
Запрос с требованием JSON возвращает валидный JSON.
Повторный retry не дублирует side effects.
Fallback при таймауте local backend работает.
II. Тесты качества маршрутизации
Правильно ли выбирается direct_local для простых вопросов.
Правильно ли выбирается local_with_rag для внутренних знаний.
Не отправляются ли общие вопросы в RAG без необходимости.
Не отправляются ли обычные запросы в cloud без причины.
Правильно ли high-risk кейсы идут на строгий путь.
Правильно ли ambiguous case отправляется в verify/escalation.
Правильно ли long-context вопрос идёт в deep route.
Правильно ли structured-output задача не идёт в слишком свободный маршрут.

Главные метрики:

route_accuracy
unnecessary_rag_rate
missed_rag_rate
unnecessary_cloud_rate
missed_cloud_rate
III. Тесты качества retrieval
Один документ, прямой ответ.
Несколько документов, ответ в одном.
Несколько документов, нужен синтез.
Документы конфликтуют.
Вопрос похож на внутренний, но ответ не в базе.
Запрос keyword-heavy.
Запрос semantic-heavy.
Запрос со смешанными синонимами.
Запрос с опечатками.
Запрос на сравнение двух версий документа.
Запрос на извлечение чисел/дат.
Запрос с ложными якорями.

Главные метрики:

retrieval_hit_rate@k
context_precision
context_recall
grounded_answer_rate
IV. Тесты локальной модели
Простая генерация.
Суммаризация.
Извлечение полей.
Классификация.
Строгий JSON.
Сложный reasoning.
Ответ на базе RAG-контекста.
Repair pass после неудачного первого ответа.
Работа на длинном контексте.
Стабильность формата на 20 повторениях.

Главные метрики:

task_pass_rate
schema_pass_rate
repair_success_rate
V. Тесты cloud escalation
Эскалация по low confidence.
Эскалация по factual conflict.
Эскалация по format failure.
Эскалация по repeated local failure.
Cloud не вызывается там, где локалка проходит.
После cloud эскалации результат действительно лучше.
Cost delta оправдан quality delta.

Главные метрики:

cloud_escalation_rate
cloud_win_rate
cost_per_successful_answer
VI. Перформанс-тесты

llama.cpp server уже даёт continuous batching и monitoring endpoints, так что эти тесты надо делать на живой нагрузке и с наблюдением backend-метрик.

Проверяй:

latency p50/p95/p99 для direct_local
latency p50/p95/p99 для local_with_rag
latency p50/p95/p99 для cloud path
throughput на 1/2/4/8 concurrent requests
queue growth
peak GPU memory
tokens/sec
impact длинного контекста
impact deep_rag
impact simultaneous users
VII. Тесты устойчивости
Local model timeout.
RAG backend timeout.
Cloud provider timeout.
Частичная деградация retrieval.
Пустой retrieval result.
Corrupted context.
Невалидный JSON от модели.
Переполнение очереди.
Restart router service.
Restart local inference service.
Потеря сети до cloud API.
Duplicate request replay.
VIII. Безопасность
Неавторизованный доступ к router API.
Неправильный API key.
Prompt injection в пользовательском запросе.
Prompt injection в retrieved context.
Попытка заставить модель проигнорировать policy.
Data exfiltration через retrieved snippets.
Логируются ли секреты.
Утечка internal route metadata в ответе пользователю.
Небезопасные заголовки и CORS.
Rate limit bypass.
Replay attack.
SSRF/unsafe URL fetch, если есть внешние инструменты.
IX. Наблюдаемость и аудит

Langfuse пригоден здесь именно потому, что даёт trace logging, transparency, cost и latency analysis в self-hosted режиме.

Проверь:

У каждого запроса есть trace id.
Видно chosen route.
Видно retrieval stats.
Видно local/cloud backend.
Видно cost.
Видно latency по шагам.
Видно final verdict.
Можно найти все cloud escalations.
Можно найти все route mistakes.
Можно сравнить две политики роутинга.
X. Regression и CI

Promptfoo удобен именно для этого: eval suites, CI/CD, self-hosting и red-teaming сценарии. DeepEval — для judge-based метрик и custom evaluation logic.

Проверь:

Ночной прогон golden dataset.
Сравнение текущего роутера с baseline.
Алерт при падении route_accuracy.
Алерт при росте unnecessary_cloud_rate.
Алерт при росте latency.
Алерт при падении schema pass rate.
Алерт при росте hallucination/groundedness failures.
Сравнение commit-to-commit.
XI. Human evaluation

Автоматических метрик недостаточно. Для 50–100 кейсов в неделю делай ручной review:

Правильность маршрута.
Был ли нужен RAG.
Было ли оправдано облако.
Насколько ответ grounded.
Была ли потеря важных деталей.
Был ли формат пригоден к использованию.
XII. Тесты выбора самого роутера

Чтобы выбрать лучшую конфигурацию, прогоняй один и тот же корпус минимум через 4 политики:

Baseline A — только local, без RAG
Baseline B — local + RAG always-on
Policy C — rules + RAG router
Policy D — rules + Semantic Router + RAG router + cloud escalation

Побеждает не тот вариант, где “самые умные ответы”, а тот, где лучший баланс:

task_pass_rate
grounded_answer_rate
cost_per_successful_answer
p95 latency
unnecessary_cloud_rate
7. Что я считаю оптимальным выбором для тебя

Я бы начал с такого стека:

FastAPI Router API
llama-server как основной inference backend
твой Cognito как RAG
Semantic Router только для pipeline routing
rule-based RAG router
rule-based cloud escalation
Langfuse
Promptfoo + DeepEval

Это даст тебе:

максимальную загрузку 3090 по полезной работе;
минимум лишних cloud-вызовов;
нормальную трассировку;
прозрачный отбор лучшей политики роутинга.


рактический PoC-план на 2 итерации под твой стек: локальный qwen3.5:27b через llama-server, твой модифицированный cognito как RAG, и ограниченное облако только для редких эскалаций.

Основа выбора такая:

FastAPI как собственный router/gateway удобен тем, что сразу даёт типизированное API и автоматическую OpenAPI-документацию, что полезно для быстрого PoC и отладки контрактов.
llama-server подходит как локальный inference-backend, потому что уже даёт OpenAI-compatible endpoints, continuous batching, monitoring endpoints и schema-constrained JSON output.
Semantic Router уместен именно как быстрый decision layer для выбора пайплайна без отдельной полной LLM-генерации на каждом шаге.
Langfuse стоит брать для self-hosted трассировки и анализа latency/cost.
Promptfoo и DeepEval хорошо дополняют друг друга: первый — для regression/eval suites и red-teaming, второй — для judge-based метрик и кастомных оценок.
Цель PoC

Понять по цифрам, а не по ощущениям:

когда локального Qwen достаточно;
когда реально нужен RAG;
когда облако оправдывает цену;
нужен ли тебе semantic router, или на старте достаточно правил.
Итерация 1 — минимально жизнеспособный router

Задача итерации:
собрать рабочий local-first контур без сложной магии, с логами и воспроизводимыми тестами.

Состав сервисов
Reverse proxy / ingress
Nginx или Caddy
auth, rate limit, IP allowlist/VPN
наружу публикуется только router API
Router API на FastAPI
входной REST API
нормализация запроса
hard rules
вызов llama-server
вызов cognito
простая эскалация в cloud
экспорт trace/event логов в Langfuse
FastAPI здесь хорош именно как быстрый typed API слой с автодокументацией.
llama-server
основной backend для локальной генерации
один основной маршрут /v1/chat/completions или /v1/responses
JSON mode для структурированных ответов
Это логично, потому что сервер уже совместим с OpenAI-style API и умеет schema-constrained JSON.
Cognito RAG
без замены
только адаптер к router API
Cloud adapter
тонкий клиент к 1–2 облачным моделям
без общего gateway
с локальным контролем бюджета
Langfuse
трассировка шагов
анализ latency/cost
хранение route decisions
Langfuse официально поддерживает self-hosting через Docker/Kubernetes/VM.
Promptfoo
ночные/ручные пакетные прогоны тест-корпуса
сравнение политик роутинга
red-team проверки RAG/роутера
Что НЕ делать в итерации 1

Не добавлять:

learned router,
bandit routing,
отдельный judge-агент,
multi-agent orchestration,
Qdrant как обязательный новый контур,
hot model swap.

Идея первой итерации — понять, работает ли local-first policy вообще.

Контракты API
1. Входной контракт router API
{
  "request_id": "uuid",
  "query": "string",
  "history_summary": "string|null",
  "attachments": [],
  "tenant": "default",
  "require_json": false,
  "risk_level": "low|medium|high",
  "budget_class": "low|normal|high",
  "source_hint": "none|internal|mixed"
}
2. Контракт решения роутера
{
  "route": "direct_local|local_with_rag|local_with_rag_and_repair|cloud_escalation",
  "rag_mode": "none|shallow|deep",
  "cloud_allowed": true,
  "reason_codes": [
    "internal_knowledge_detected",
    "format_strict",
    "high_risk",
    "retrieval_low_confidence"
  ]
}
3. Контракт ответа модели
{
  "answer": "string",
  "citations": [],
  "confidence": 0.0,
  "grounded": true,
  "used_context": true,
  "repair_applied": false,
  "escalated_to_cloud": false
}
Логика роутинга в итерации 1

Только правила.

Router A — нужен ли RAG

Стартовая эвристика:

Отправляй в local_with_rag, если есть хотя бы одно:

запрос ссылается на внутренние документы/данные/прошлые обсуждения;
есть слова-маркеры: “по документу”, “в базе”, “в проекте”, “найди расхождения”, “сравни версии”, “что у нас написано”;
source_hint=internal|mixed;
вопрос выглядит как knowledge lookup, а не как general reasoning.

Иначе — direct_local.

Router B — shallow или deep RAG

Стартовая эвристика:

shallow_rag, если:

нужно найти 1 факт / 1 документ / 1 сущность;
короткий вопрос;
ожидается прямой ответ.

deep_rag, если:

“сравни”,
“сопоставь”,
“в нескольких документах”,
“найди противоречия”,
“сделай итог по всем материалам”.
Router C — нужна ли cloud escalation

На первой итерации эскалация только по жёстким условиям:

risk_level=high;
локальный ответ не прошёл JSON/schema check;
retrieval дал слабый контекст;
локальный ответ противоречит найденному контексту;
после 1 repair-попытки ответ всё ещё плохой.
Стартовые пороги для PoC

Это мои рекомендуемые стартовые эвристики, их потом надо калибровать по логам.

Для RAG
top1_score >= 0.78 и top1-top2 >= 0.08 → shallow_rag
0.60 <= top1_score < 0.78 → deep_rag
top1_score < 0.60 → либо direct_local, либо cloud_escalation, в зависимости от risk level
Для repair

Запускать 1 локальный repair-pass, если:

ответ пустой или слишком общий;
ответ не соответствует формату;
не использованы найденные документы;
confidence самого пайплайна низкий.
Для cloud

Эскалировать в cloud, если:

risk_level=high
либо deep_rag + слабый retrieval
либо локальный repair не помог
либо задача явно требует максимальной точности и цена ошибки выше бюджета
Что именно поднять первым

Порядок развертывания:

llama-server
Router API
адаптер к Cognito
Langfuse
минимальный cloud adapter
Promptfoo test suite
Что должно заработать к концу итерации 1
запрос приходит в Router API;
роутер выбирает direct_local или local_with_rag;
ответ возвращается единым форматом;
каждый запрос оставляет trace;
можно прогнать пакет тестов и сравнить 3 политики:
local only
local + rag always-on
rules-based routing
Итерация 2 — semantic routing и нормальная оценка качества

Задача итерации:
проверить, даёт ли Semantic Router и более аккуратная эскалация реальную экономию и прирост качества.

Что добавляется
1. Semantic Router

Встраиваешь его только в один узкий слой:
выбор пайплайна, а не полный контроль всей системы.
Это соответствует его назначению как быстрого semantic decision layer.

Маршруты:

direct_local
local_with_rag
local_with_rag_and_repair
cloud_candidate
2. Local verifier

Не отдельный агент, а короткая локальная проверка тем же Qwen:

ответ опирается на контекст?
формат корректен?
есть ли явные пробелы?
был ли дан ответ на вопрос?
3. DeepEval

Вводишь judge-based метрики:

answer relevancy
groundedness/custom GEval
route correctness
DeepEval как раз даёт готовые и кастомные LLM-based метрики через G-Eval.
4. Promptfoo red team

Запускаешь наборы против:

prompt injection,
RAG poisoning,
data exfiltration hints,
route confusion.
Promptfoo отдельно покрывает red teaming и RAG-oriented security testing.
Что сравнивать во 2 итерации

Прогоняешь одинаковый корпус через 4 политики:

Policy A

local only

Policy B

local + rag always-on

Policy C

rules router + rag router + local repair

Policy D

rules + semantic router + rag router + local repair + cloud escalation

Побеждает не самая “умная”, а та, у которой лучше баланс:

task_pass_rate
grounded_pass_rate
cost_per_successful_answer
p95_latency
unnecessary_rag_rate
unnecessary_cloud_rate
Минимальный golden dataset

Для PoC достаточно 120–180 кейсов.

Разбивка:

General / без RAG — 30
Прямые internal lookup — 30
Сложный RAG / multi-doc synthesis — 25
Строгий JSON / extraction — 20
High-risk / точность важна — 15
Adversarial / injection / route confusion — 20
Пограничные ambiguous кейсы — 20

Это уже даст тебе статистически полезную картину, даже без большого production-трафика.

Полный список тестов для PoC
Блок 1. Корректность маршрута
Общий вопрос идёт в direct_local
Внутренний вопрос идёт в local_with_rag
Multi-doc вопрос идёт в deep_rag
High-risk вопрос идёт в cloud_candidate
JSON-задача идёт в structured path
Ambiguous запрос не уходит сразу в облако
RAG не включается без нужды
Cloud не включается без нужды
Блок 2. Retrieval
Есть 1 релевантный документ
Есть 2–3 близких документа
Нужен синтез по нескольким документам
Документы конфликтуют
Нет релевантных документов
Запрос с опечатками
Запрос с ключевыми словами
Запрос с синонимами
Запрос на даты/числа
Запрос на различия версий
Блок 3. Генерация локальной модели
Прямой ответ
Суммаризация
Извлечение полей
Строгий JSON
Ответ по RAG-контексту
Repair после плохого ответа
Длинный контекст
10 повторов на стабильность формата
Блок 4. Эскалация в облако
High-risk кейс сразу эскалируется
Локальный repair спасает без облака
Формат-фейл эскалируется
Слабый retrieval эскалируется
Cloud реально улучшает ответ
Cloud не вызывается там, где local уже хорош
Блок 5. Надёжность
timeout локальной модели
timeout RAG
timeout cloud
пустой retrieval
невалидный JSON
повторный запрос с тем же request_id
частичный сбой Langfuse
перегрузка очереди
Блок 6. Безопасность
prompt injection в user query
prompt injection внутри retrieved context
попытка вытащить системный промпт
route confusion
data exfiltration hints
подмена role/system-like текста
invalid API token
rate-limit abuse
Какие метрики считать с первого дня
Основные
route_accuracy
task_pass_rate
grounded_pass_rate
cost_per_successful_answer
p95_latency
Вторичные
unnecessary_rag_rate
missed_rag_rate
unnecessary_cloud_rate
repair_success_rate
schema_pass_rate
retrieval_hit_rate@k
Как принять решение после PoC
Оставлять только rules-based router, если:
Policy C почти не хуже Policy D
экономия от semantic router минимальна
latency выросла, а качество почти нет
Добавлять semantic router в прод, если:
Policy D заметно снижает unnecessary_rag_rate
и/или снижает unnecessary_cloud_rate
и не ухудшает task_pass_rate
Усиливать cloud usage, если:
high-risk кейсы стабильно проваливаются локально
local repair почти не помогает
прирост качества от cloud выше его стоимости
Не усиливать cloud usage, если:
локалка после repair уже проходит большинство кейсов
cloud выигрывает редко
цена успеха выходит слишком высокой
Что я считаю лучшим стартовым вариантом

Для тебя я бы запускал PoC так:

Этап 1
FastAPI router
llama-server
Cognito RAG
Langfuse
Promptfoo
rules-based routing
Этап 2
добавить Semantic Router
добавить DeepEval
включить локальный verifier
включить ограниченную cloud escalation

Это даст максимально контролируемую систему без LiteLLM и без ненужной сложности, но с нормальным путём к измеримому улучшению. FastAPI и llama-server упрощают интеграцию, Semantic Router закрывает узкий слой выбора пайплайна, а Langfuse/Promptfoo/DeepEval дают наблюдаемость и eval-контур.
