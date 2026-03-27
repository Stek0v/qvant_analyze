"""База знаний GPU и моделей для автоконфигурации qvant."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    vram_mb: int
    compute_cap: float
    bandwidth_gbps: int
    supports_flash: bool  # CC >= 7.5


@dataclass(frozen=True)
class ModelSpec:
    family: str
    params_b: float
    quant: str
    vram_mb: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    native_ctx: int
    hf_repo: str
    gguf_filename: str
    supports_thinking: bool

    @property
    def display_name(self) -> str:
        return f"{self.family}:{self.params_b:.0f}b"


# =====================================================================
# GPU Database
# =====================================================================

GPU_DB: dict[str, GPUSpec] = {
    # Consumer — Ampere
    "RTX 3060":  GPUSpec("RTX 3060",  12288, 8.6,  360, True),
    "RTX 3070":  GPUSpec("RTX 3070",   8192, 8.6,  448, True),
    "RTX 3080":  GPUSpec("RTX 3080",  10240, 8.6,  760, True),
    "RTX 3090":  GPUSpec("RTX 3090",  24576, 8.6,  936, True),
    # Consumer — Ada
    "RTX 4060":  GPUSpec("RTX 4060",   8192, 8.9,  272, True),
    "RTX 4070":  GPUSpec("RTX 4070",  12288, 8.9,  504, True),
    "RTX 4080":  GPUSpec("RTX 4080",  16384, 8.9,  717, True),
    "RTX 4090":  GPUSpec("RTX 4090",  24576, 8.9, 1008, True),
    # Consumer — Blackwell
    "RTX 5090":  GPUSpec("RTX 5090",  32768, 10.0, 1792, True),
    # Datacenter
    "T4":        GPUSpec("T4",        16384, 7.5,  300, True),
    "A6000":     GPUSpec("A6000",     49152, 8.6,  768, True),
    "L40S":      GPUSpec("L40S",      49152, 8.9,  864, True),
    "A100-40":   GPUSpec("A100 40GB", 40960, 8.0, 1555, True),
    "A100-80":   GPUSpec("A100 80GB", 81920, 8.0, 2039, True),
    "H100":      GPUSpec("H100",      81920, 9.0, 3350, True),
}

# =====================================================================
# Model Database (Q4_K_M quantization)
# =====================================================================

MODEL_DB: dict[str, ModelSpec] = {
    "llama3.1:8b": ModelSpec(
        "llama3.1", 8, "Q4_K_M", 4300, 32, 8, 128, 131072,
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", False,
    ),
    "llama3.1:70b": ModelSpec(
        "llama3.1", 70, "Q4_K_M", 39000, 80, 8, 128, 131072,
        "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf", False,
    ),
    "mistral:7b": ModelSpec(
        "mistral", 7, "Q4_K_M", 4500, 32, 8, 128, 32768,
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", False,
    ),
    "mixtral:8x7b": ModelSpec(
        "mixtral", 46.7, "Q4_K_M", 22000, 32, 8, 128, 32768,
        "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf", False,
    ),
    "qwen3.5:7b": ModelSpec(
        "qwen3.5", 7, "Q4_K_M", 4500, 28, 4, 128, 131072,
        "bartowski/Qwen_Qwen3.5-7B-GGUF",
        "Qwen_Qwen3.5-7B-Q4_K_M.gguf", True,
    ),
    "qwen3.5:14b": ModelSpec(
        "qwen3.5", 14, "Q4_K_M", 9000, 48, 4, 128, 131072,
        "bartowski/Qwen_Qwen3.5-14B-GGUF",
        "Qwen_Qwen3.5-14B-Q4_K_M.gguf", True,
    ),
    "qwen3.5:27b": ModelSpec(
        "qwen3.5", 27, "Q4_K_M", 17000, 64, 4, 128, 131072,
        "bartowski/Qwen_Qwen3.5-27B-GGUF",
        "Qwen_Qwen3.5-27B-Q4_K_M.gguf", True,
    ),
    "phi-4:14b": ModelSpec(
        "phi-4", 14, "Q4_K_M", 8000, 40, 10, 96, 16384,
        "bartowski/phi-4-GGUF",
        "phi-4-Q4_K_M.gguf", False,
    ),
    "gemma-3:12b": ModelSpec(
        "gemma-3", 12, "Q4_K_M", 6600, 34, 8, 256, 131072,
        "bartowski/gemma-3-12b-it-GGUF",
        "gemma-3-12b-it-Q4_K_M.gguf", False,
    ),
}

# =====================================================================
# KV cache bytes per element
# =====================================================================

KV_BYTES: dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
}

OVERHEAD_MB = 1100  # Desktop + driver + runtime


# =====================================================================
# Detection
# =====================================================================

def detect_gpu() -> GPUSpec:
    """Определить GPU через nvidia-smi. Fallback на raw данные."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        line = result.stdout.strip().split("\n")[0]
        name_raw, vram_str = [s.strip() for s in line.split(",")]
        vram_mb = int(vram_str)

        # Попробовать найти в базе
        for key, spec in GPU_DB.items():
            if key.lower() in name_raw.lower():
                # Вернуть с реальным VRAM (может отличаться от базы)
                return GPUSpec(
                    name=spec.name, vram_mb=vram_mb,
                    compute_cap=spec.compute_cap,
                    bandwidth_gbps=spec.bandwidth_gbps,
                    supports_flash=spec.supports_flash,
                )

        # Не в базе — fallback с консервативными значениями
        return GPUSpec(
            name=name_raw, vram_mb=vram_mb,
            compute_cap=7.0, bandwidth_gbps=0,
            supports_flash=True,  # Большинство современных карт поддерживают
        )
    except Exception as e:
        raise RuntimeError(f"Не удалось определить GPU: {e}")


def get_model_spec(model_name: str) -> ModelSpec:
    """Получить спецификацию модели по имени."""
    if model_name in MODEL_DB:
        return MODEL_DB[model_name]
    # Попробовать нечёткий поиск
    name_lower = model_name.lower()
    for key, spec in MODEL_DB.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return spec
    raise KeyError(f"Модель '{model_name}' не найдена в базе. Доступные: {list(MODEL_DB.keys())}")


# =====================================================================
# Calculations
# =====================================================================

def kv_bytes_per_token(model: ModelSpec, cache_type: str = "q8_0") -> float:
    """Байт KV cache на один токен контекста."""
    bpe = KV_BYTES.get(cache_type, 1.0)
    # 2 (K + V) * layers * kv_heads * head_dim * bytes_per_element
    return 2 * model.num_layers * model.num_kv_heads * model.head_dim * bpe


def calculate_max_context(
    gpu: GPUSpec,
    model: ModelSpec,
    cache_type: str = "q8_0",
    overhead_mb: int = OVERHEAD_MB,
) -> int:
    """Максимальный контекст, влезающий в VRAM."""
    available_bytes = (gpu.vram_mb - model.vram_mb - overhead_mb) * 1024 * 1024
    if available_bytes <= 0:
        return 0
    bpt = kv_bytes_per_token(model, cache_type)
    if bpt <= 0:
        return 0
    max_ctx = int(available_bytes / bpt)
    # Округлить вниз до 1024 и ограничить native контекстом
    max_ctx = (max_ctx // 1024) * 1024
    return min(max_ctx, model.native_ctx)


def get_context_tiers(
    gpu: GPUSpec,
    model: ModelSpec,
    cache_type: str = "q8_0",
) -> dict[str, list[int]]:
    """Рассчитать тиры контекста: core / extended / max."""
    max_ctx = calculate_max_context(gpu, model, cache_type)
    tiers: dict[str, list[int]] = {"core": [], "extended": [], "max": []}

    for ctx in [2048, 4096, 8192, 16384, 24576, 32768, 65536, 131072]:
        if ctx > max_ctx:
            break
        if ctx <= max_ctx * 0.4:
            tiers["core"].append(ctx)
        elif ctx <= max_ctx * 0.7:
            tiers["extended"].append(ctx)
        else:
            tiers["max"].append(ctx)

    # Гарантировать хотя бы 1 core entry
    if not tiers["core"] and max_ctx >= 2048:
        tiers["core"] = [2048]

    return tiers


def recommend_models(gpu: GPUSpec, min_ctx: int = 2048) -> list[tuple[ModelSpec, int]]:
    """Рекомендовать модели под GPU, отсортированные по params (качеству).

    Возвращает [(model, max_context)] — модели что влезают с минимум min_ctx.
    """
    results = []
    for model in MODEL_DB.values():
        max_ctx = calculate_max_context(gpu, model)
        if max_ctx >= min_ctx:
            results.append((model, max_ctx))
    # Сортируем: больше параметров → выше качество → первые
    results.sort(key=lambda x: x[0].params_b, reverse=True)
    return results


def model_fits(gpu: GPUSpec, model: ModelSpec) -> bool:
    """Влезает ли модель в GPU с минимальным контекстом."""
    return calculate_max_context(gpu, model) >= 2048
