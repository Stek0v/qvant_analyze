"""Центральная конфигурация проекта qvant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Пути ---
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
REPORTS_DIR = RESULTS_DIR / "reports"

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_DEFAULT = os.getenv("QVANT_MODEL", "qwen3.5:27b")
MODEL_CUSTOM = "qwen3.5-27b-optimized"

# max_tokens для разных think modes (llama-server)
MAX_TOKENS_NO_THINK = 1024
MAX_TOKENS_THINK = 4096

# --- GPU (авто-детект с fallback на env var) ---
def _detect_gpu_vram() -> int:
    try:
        from gpu_profiles import detect_gpu
        return detect_gpu().vram_mb
    except Exception:
        return int(os.getenv("QVANT_GPU_VRAM_MB", "24576"))

GPU_TOTAL_VRAM_MB = _detect_gpu_vram()
GPU_OVERHEAD_MB = int(os.getenv("QVANT_GPU_OVERHEAD_MB", "1100"))

# --- Таймауты и пороги ---
TEST_TIMEOUT_SECONDS = 300  # 5 min max per inference
COOLDOWN_SECONDS = 3        # пауза между прогонами
WARMUP_PROMPT = "Привет, ответь одним словом: работаешь?"
HEALTH_CHECK_RETRIES = 30
HEALTH_CHECK_INTERVAL = 1.0

# --- Целевые диапазоны токенов для оценки conciseness ---
TOKEN_TARGETS = {
    "A": (50, 200),
    "B": (200, 600),
    "C": (500, 2000),
}


@dataclass(frozen=True)
class TestConfig:
    """Одна конфигурация для тестирования."""
    name: str
    model: str = MODEL_DEFAULT
    kv_cache_type: str = "q8_0"
    num_ctx: int = 8192
    think: Any = False          # False, "low", "medium", "high"
    temperature: float = 0.2
    top_p: float = 0.95
    presence_penalty: float | None = None  # None = модельный дефолт


@dataclass
class BenchmarkResult:
    """Результат одного вызова chat."""
    # Идентификация
    run_id: str = ""
    timestamp: str = ""
    config_name: str = ""
    prompt_set: str = ""
    prompt_index: int = 0
    prompt_text: str = ""

    # Конфигурация
    model: str = ""
    kv_cache_type: str = ""
    num_ctx: int = 0
    think_mode: Any = False
    temperature: float = 0.0

    # Метрики Ollama (мс)
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    prompt_eval_count: int = 0
    prompt_eval_duration_ms: float = 0.0
    eval_count: int = 0
    eval_duration_ms: float = 0.0

    # Вычисляемые
    tokens_per_sec: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    is_cold_start: bool = False

    # GPU
    gpu_vram_pre_mb: int = 0
    gpu_vram_post_mb: int = 0
    gpu_vram_peak_mb: int = 0
    gpu_util_avg_pct: float = 0.0
    gpu_temp_max_c: int = 0
    gpu_processor: str = ""

    # Ответ
    content: str = ""
    thinking: str | None = None
    thinking_token_estimate: int = 0
    content_token_estimate: int = 0

    # Качество (автоматическое)
    quality_completeness: float = 0.0
    quality_structure: float = 0.0
    quality_conciseness: float = 0.0
    quality_composite: float = 0.0

    # Качество (ручное, заполняется позже)
    manual_accuracy: int | None = None
    manual_completeness: int | None = None
    manual_reasoning: int | None = None
    manual_usefulness: int | None = None


# --- Определение фаз тестирования ---

def _auto_context_tiers() -> dict[str, list[int]]:
    """Рассчитать тиры контекста для текущего GPU + модели."""
    try:
        from gpu_profiles import detect_gpu, get_model_spec, get_context_tiers as _get_tiers
        gpu = detect_gpu()
        model = get_model_spec(MODEL_DEFAULT)
        return _get_tiers(gpu, model)
    except Exception:
        return {"core": [4096, 8192, 16384], "extended": [24576, 32768], "max": [65536]}


def build_phase1_configs(
    model: str | None = None,
    ctx_sizes: list[int] | None = None,
) -> list[TestConfig]:
    """Core matrix: q8_0 KV, динамические ctx, 3 think, temp=0.2."""
    model = model or MODEL_DEFAULT
    if ctx_sizes is None:
        ctx_sizes = _auto_context_tiers()["core"][:3]
        if not ctx_sizes:
            ctx_sizes = [2048, 4096]
    configs = []
    for ctx in ctx_sizes:
        for think in (False, "low", "medium"):
            think_label = str(think) if think else "off"
            name = f"q8kv_ctx{ctx // 1024}k_think-{think_label}_t02"
            configs.append(TestConfig(
                name=name, model=model, num_ctx=ctx, think=think, temperature=0.2,
            ))
    return configs


def build_phase2_configs(
    model: str | None = None,
    ctx_sizes: list[int] | None = None,
) -> list[TestConfig]:
    """KV cache comparison: f16 и q4_0."""
    model = model or MODEL_DEFAULT
    if ctx_sizes is None:
        tiers = _auto_context_tiers()
        ctx_sizes = tiers["core"][-2:]  # последние 2 core
        if not ctx_sizes:
            ctx_sizes = [4096]
    configs = []
    for kv in ("f16", "q4_0"):
        for ctx in ctx_sizes:
            for think in (False, "medium"):
                think_label = str(think) if think else "off"
                name = f"{kv}kv_ctx{ctx // 1024}k_think-{think_label}_t02"
                configs.append(TestConfig(
                    name=name, model=model, kv_cache_type=kv, num_ctx=ctx,
                    think=think, temperature=0.2,
                ))
    return configs


def build_phase3_configs(
    model: str | None = None,
    ctx_sizes: list[int] | None = None,
) -> list[TestConfig]:
    """Extended context: все think modes."""
    model = model or MODEL_DEFAULT
    if ctx_sizes is None:
        tiers = _auto_context_tiers()
        ctx_sizes = tiers["extended"] + tiers["max"]
        if not ctx_sizes:
            return []  # нет места для extended
    configs = []
    for ctx in ctx_sizes:
        for think in (False, "low", "medium", "high"):
            think_label = str(think) if think else "off"
            name = f"q8kv_ctx{ctx // 1024}k_think-{think_label}_t02"
            configs.append(TestConfig(
                name=name, model=model, num_ctx=ctx, think=think, temperature=0.2,
            ))
    return configs


def build_phase4_configs(best_ctx: int = 8192, best_think: Any = False) -> list[TestConfig]:
    """Temperature sweep на лучшем конфиге."""
    configs = []
    for temp in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        name = f"q8kv_ctx{best_ctx // 1024}k_think-{best_think or 'off'}_t{temp:.2f}"
        configs.append(TestConfig(
            name=name, num_ctx=best_ctx, think=best_think, temperature=temp,
        ))
    return configs
