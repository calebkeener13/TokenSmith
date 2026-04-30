"""
Runtime benchmarking for the TokenSmith RAG pipeline.

Measures per-stage latency, end-to-end latency, generation throughput
(tokens/sec), and peak memory consumption. Results are written as JSON
to the logs/ directory and can be printed as a one-line summary.

Usage:
    with BenchmarkTimer() as bm:
        bm.start("retrieval")
        ...
        bm.stop("retrieval")

        bm.start("rerank")
        ...
        bm.stop("rerank")

        bm.start("generation")
        tokens = sum(1 for _ in stream_iter)
        bm.stop("generation")

    bm.finalize(token_count=tokens)
    bm.save("logs/")
    print(bm.summary())
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _peak_memory_mb() -> float:
    """Return current process RSS in MB. Returns 0.0 on unsupported platforms."""
    try:
        import resource
        # RUSAGE_SELF returns bytes on Linux, pages on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return usage / (1024 * 1024)  # bytes → MB
        else:
            return usage / 1024           # KB → MB
    except Exception:
        return 0.0


class BenchmarkTimer:
    """
    Context manager that times named pipeline stages and collects memory/throughput.

    Enter the context to start the wall-clock timer and capture the baseline
    memory footprint. Use start()/stop() pairs inside the block to time
    individual stages. Call finalize() before exiting (or after) to record
    the token count. Call save() to persist results; summary() for a one-liner.
    """

    def __init__(self):
        self._wall_start: float = 0.0
        self._wall_end: float = 0.0
        self._mem_start_mb: float = 0.0
        self._stage_starts: Dict[str, float] = {}
        self.stages_sec: Dict[str, float] = {}
        self.total_sec: float = 0.0
        self.tokens_generated: int = 0
        self.tokens_per_sec: float = 0.0
        self.peak_memory_mb: float = 0.0
        self._backend: str = ""

    def begin(self) -> "BenchmarkTimer":
        """Start the overall timer. Call this when not using as a context manager."""
        self._wall_start = time.perf_counter()
        self._mem_start_mb = _peak_memory_mb()
        try:
            from src.hardware import detect_backend
            hw = detect_backend()
            self._backend = hw.backend_name
        except Exception:
            self._backend = "unknown"
        return self

    def end(self) -> None:
        """Stop the overall timer. Call this when not using as a context manager."""
        self._wall_end = time.perf_counter()
        self.total_sec = self._wall_end - self._wall_start
        mem_end = _peak_memory_mb()
        self.peak_memory_mb = max(0.0, mem_end - self._mem_start_mb)

    def __enter__(self) -> "BenchmarkTimer":
        return self.begin()

    def __exit__(self, *_):
        self.end()

    def start(self, stage: str) -> None:
        self._stage_starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """Stop timing a stage. Returns elapsed seconds for that stage."""
        if stage not in self._stage_starts:
            raise KeyError(f"BenchmarkTimer: stop('{stage}') called without a matching start()")
        elapsed = time.perf_counter() - self._stage_starts.pop(stage)
        self.stages_sec[stage] = self.stages_sec.get(stage, 0.0) + elapsed
        return elapsed

    def finalize(self, token_count: int) -> None:
        """Record the number of tokens generated and compute throughput."""
        self.tokens_generated = token_count
        gen_sec = self.stages_sec.get("generation", 0.0)
        self.tokens_per_sec = token_count / gen_sec if gen_sec > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "platform": f"{platform.system()} {platform.machine()}",
            "backend": self._backend,
            "stages_sec": {k: round(v, 4) for k, v in self.stages_sec.items()},
            "total_sec": round(self.total_sec, 4),
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }

    def save(self, logs_dir: str = "logs") -> Path:
        """Write results as JSON. Returns the path written."""
        out_dir = Path(logs_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"benchmark_{timestamp_str}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
        return out_path

    def summary(self) -> str:
        """Return a single-line human-readable summary."""
        stage_parts = ", ".join(
            f"{k}={v:.2f}s" for k, v in self.stages_sec.items()
        )
        return (
            f"[Benchmark] total={self.total_sec:.2f}s | "
            f"{stage_parts} | "
            f"{self.tokens_per_sec:.1f} tok/s | "
            f"mem_delta={self.peak_memory_mb:.1f} MB | "
            f"backend={self._backend}"
        )
