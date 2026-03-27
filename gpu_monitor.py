"""Мониторинг GPU через nvidia-smi subprocess."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


@dataclass
class GPUSnapshot:
    vram_used_mb: int
    vram_free_mb: int
    vram_total_mb: int
    gpu_util_pct: int
    temp_c: int
    power_w: float
    timestamp: str


def snapshot() -> GPUSnapshot:
    """Одиночный снимок состояния GPU."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    parts = [p.strip() for p in result.stdout.strip().split(",")]

    return GPUSnapshot(
        vram_used_mb=int(parts[0]),
        vram_free_mb=int(parts[1]),
        vram_total_mb=int(parts[2]),
        gpu_util_pct=int(parts[3]),
        temp_c=int(parts[4]),
        power_w=float(parts[5]),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


class GPUMonitor:
    """Фоновый сборщик снимков GPU во время выполнения callback."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval

    def monitor_during(self, callback: Callable) -> tuple[any, list[GPUSnapshot]]:
        """Выполнить callback, собирая снимки GPU в фоне.

        Возвращает (результат callback, список снимков).
        """
        snapshots: list[GPUSnapshot] = []
        stop_event = threading.Event()

        def _collector():
            while not stop_event.is_set():
                try:
                    snapshots.append(snapshot())
                except Exception:
                    pass
                stop_event.wait(self.interval)

        thread = threading.Thread(target=_collector, daemon=True)
        thread.start()
        try:
            result = callback()
        finally:
            stop_event.set()
            thread.join(timeout=2)

        return result, snapshots

    @staticmethod
    def summarize(snapshots: list[GPUSnapshot]) -> dict:
        """Агрегировать снимки: peak VRAM, avg util, max temp."""
        if not snapshots:
            return {
                "peak_vram_mb": 0,
                "avg_util_pct": 0.0,
                "max_temp_c": 0,
                "max_power_w": 0.0,
                "snapshot_count": 0,
            }
        return {
            "peak_vram_mb": max(s.vram_used_mb for s in snapshots),
            "avg_util_pct": sum(s.gpu_util_pct for s in snapshots) / len(snapshots),
            "max_temp_c": max(s.temp_c for s in snapshots),
            "max_power_w": max(s.power_w for s in snapshots),
            "snapshot_count": len(snapshots),
        }
