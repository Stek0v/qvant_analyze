LiteLLM: что тестируют
В репозитории LiteLLM есть локальные тесты для least-busy и lowest-latency роутинга, где проверяется, что роутер выбирает нужный deployment по состоянию очереди или по измеренной задержке. Эти тесты хороши именно для проверки корректности логики роутинга на уровне поведения API/стратегии, а не качества ответа модели.

Также в документации LiteLLM описаны стратегии routing/load balancing вроде simple-shuffle, latency-based-routing, usage-based-routing и pre-call checks, что помогает понять, какие сценарии стоит покрывать собственными тестами.

Готовые датасеты
Самый полезный готовый датасет для самостоятельного тестирования роутеров — RouterBench. В статье и репозитории указано, что он содержит более 405k samples, охватывает 8 задач/датасетов и 11 моделей, включая commonsense, MMLU, MT-Bench, GSM8K, MBPP и RAG-часть.

Отдельно у RouterBench есть 657 RAG-пар вопросов-ответов, где вопросы собраны из news и Wikipedia; это удобно, если хочешь тестировать роутинг между моделями с и без internet/retrieval capability.

Что именно можно брать
Ниже — наиболее практичные наборы для self-testing роутеров:

Источник	Что внутри	Для чего полезно
LiteLLM tests	least-busy, lowest-latency и связанные unit-tests	Проверка, что роутер выбирает правильный deployment 
RouterBench	405k+ samples, 8 benchmark datasets, 11 models	Сравнение стратегий роутинга по cost/performance 
RouterBench RAG subset	657 QA-пар по news/Wikipedia	Тест роутинга для retrieval-aware сценариев 
Semantic router datasets	синтетические датасеты для semantic routing	Если нужен роутинг “tool vs direct answer” или intent routing 
Как использовать
Если тебе нужен именно regression test для LiteLLM, бери идеи из test_least_busy_routing.py и test_lowest_latency_routing.py: там удобно мокать/подставлять метрики и проверять, что выбран правильный model_id или deployment.

Если нужен более академичный/бенчмарочный подход, бери RouterBench и прогоняй свои роутер-стратегии на тех же inputs, сравнивая accuracy/cost/latency и строя own oracle baseline. Репозиторий RouterBench прямо описывает pipeline convert_data.py -> evaluate_routers.py -> visualize_results.py.

LiteLLM тесты
Ключевые тестовые файлы LiteLLM, которые прямо проверяют роутинг: tests/local_testing/test_least_busy_routing.py и tests/local_testing/test_lowest_latency_routing.py. Первый покрывает выбор least-busy deployment, второй — выбор deployment с минимальной latency, включая сценарии с таймаутами, первым выбором и latency buffer.

Полезно также смотреть сам litellm/router.py, потому что там находится основная логика Router, а документация по routing/load-balancing перечисляет стратегии simple-shuffle, least-busy, usage-based-routing, latency-based-routing, cost-based-routing.

Что именно брать
tests/local_testing/test_least_busy_routing.py — для проверки, что выбирается наименее загруженный deployment.

tests/local_testing/test_lowest_latency_routing.py — для проверки latency-based routing и fallback-поведения.

litellm/router.py — как reference для того, какие поля и стратегия вообще участвуют в выборе маршрута.

docs/routing-load-balancing и docs/proxy/load_balancing — как source of truth по стратегиям и ожидаемому поведению.

Датасеты
RouterBench доступен на Hugging Face как withmartian/routerbench. В карточке датасета указано, что это более 30k prompts с ответами от 11 LLM, а в обзорах/странице бенчмарка — что весь набор содержит 405k+ inference outcomes, 8 datasets и 11 models, включая commonsense, MMLU, MT-Bench, GSM8K, MBPP и RAG.

Прямая ссылка: https://huggingface.co/datasets/withmartian/routerbench. У RouterBench также есть raw large file routerbench_raw.pkl на HF, а обсуждение dataset card отдельно упоминает две версии: 5-shot и 0-shot.

Для semantic routing есть tai-tai-sama/semantic-router-dataset на Hugging Face; это synthetic dataset для fine-tuning semantic routers, где задача — маршрутизировать запрос к tool или давать direct answer, если tool не нужен. У датасета есть parquet-файлы и отдельная директория data.

Ссылки
RouterBench: https://huggingface.co/datasets/withmartian/routerbench

RouterBench raw file: https://huggingface.co/datasets/withmartian/routerbench/blob/main/routerbench_raw.pkl

Semantic router dataset: https://huggingface.co/datasets/tai-tai-sama/semantic-router-dataset

Semantic router data tree: https://huggingface.co/datasets/tai-tai-sama/semantic-router-dataset/tree/main/data

Шаблон pytest
Ниже — минимальный, но практичный шаблон, который можно адаптировать под LiteLLM Router, OpenRouter-подобный gateway или свой OpenAI-compatible router.

python
import pytest
from collections import defaultdict

class FakeDeployment:
    def __init__(self, name, latency=0.0, busy=0, cost=0.0):
        self.name = name
        self.latency = latency
        self.busy = busy
        self.cost = cost

class FakeRouter:
    def __init__(self, deployments):
        self.deployments = deployments

    def route_least_busy(self):
        return min(self.deployments, key=lambda d: d.busy).name

    def route_lowest_latency(self):
        return min(self.deployments, key=lambda d: d.latency).name

    def route_lowest_cost(self):
        return min(self.deployments, key=lambda d: d.cost).name


@pytest.fixture
def router():
    deployments = [
        FakeDeployment("a", latency=120, busy=3, cost=0.01),
        FakeDeployment("b", latency=80, busy=1, cost=0.02),
        FakeDeployment("c", latency=150, busy=0, cost=0.005),
    ]
    return FakeRouter(deployments)


def test_least_busy(router):
    assert router.route_least_busy() == "c"


def test_lowest_latency(router):
    assert router.route_lowest_latency() == "b"


def test_lowest_cost(router):
    assert router.route_lowest_cost() == "c"
Более близко к LiteLLM
Если хочешь проверять именно LiteLLM-подобный контракт, я бы делал так:

python
import pytest

@pytest.mark.parametrize(
    "deployments, expected",
    [
        ([{"id": "d1", "busy": 10}, {"id": "d2", "busy": 2}], "d2"),
        ([{"id": "d1", "busy": 0}, {"id": "d2", "busy": 0}], "d1"),
    ],
)
def test_route_least_busy_contract(deployments, expected):
    chosen = min(deployments, key=lambda d: d["busy"])["id"]
    assert chosen == expected
И отдельно держать тесты на:

tie-break behavior,

cache warmup,

timeout/fallback,

rate-limit exhaustion,

deterministic seed for shuffle,

latency window / TTL decay.

Рекомендованный набор для локального прогона
Я бы стартовал с такого минимума:

5–10 unit tests на pure routing logic.

3–5 integration tests с мокнутыми endpoint latency/busy/cost.

1 dataset-driven eval на RouterBench.

1 semantic-routing eval на semantic-router-dataset.

