"""Observability - Prometheus metrics and structured logging."""

import logging
import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

# ============================================================
# Prometheus metrics
# ============================================================

INFERENCE_REQUESTS = Counter(
    "ai_tpm_inference_total",
    "Total inference requests",
    ["model", "status"],
)

INFERENCE_LATENCY = Histogram(
    "ai_tpm_inference_latency_seconds",
    "Inference latency in seconds",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

TTFT_HISTOGRAM = Histogram(
    "ai_tpm_ttft_seconds",
    "Time to first token",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

TPOT_HISTOGRAM = Histogram(
    "ai_tpm_tpot_seconds",
    "Time per output token",
    ["model"],
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
)

TPS_GAUGE = Gauge(
    "ai_tpm_tokens_per_second",
    "Current tokens per second",
    ["model"],
)

GPU_UTILIZATION = Gauge(
    "ai_tpm_gpu_utilization_percent",
    "GPU utilization percentage",
    ["device"],
)

CPU_UTILIZATION = Gauge(
    "ai_tpm_cpu_utilization_percent",
    "CPU utilization percentage",
)

RAM_UTILIZATION = Gauge(
    "ai_tpm_ram_utilization_percent",
    "RAM utilization percentage",
)

BENCHMARK_RUNS = Counter(
    "ai_tpm_benchmark_runs_total",
    "Total benchmark runs",
    ["model"],
)

READINESS_SCORE = Gauge(
    "ai_tpm_readiness_score",
    "GPU readiness score percentage",
    ["model", "dimension"],
)


# ============================================================
# Structured logger
# ============================================================

def get_logger(name: str = "ai_tpm") -> logging.Logger:
    """Get a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


# ============================================================
# Timing context manager
# ============================================================

@contextmanager
def timed_operation(name: str) -> Generator[dict, None, None]:
    """Context manager that times an operation and records metrics."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
        logger.info(f"{name} completed in {result['elapsed']:.3f}s")


def record_inference_metrics(
    model: str,
    latency: float,
    ttft: float,
    tpot: float,
    tps: float,
    success: bool,
):
    """Record inference metrics to Prometheus."""
    status = "success" if success else "error"
    INFERENCE_REQUESTS.labels(model=model, status=status).inc()
    INFERENCE_LATENCY.labels(model=model).observe(latency)
    TTFT_HISTOGRAM.labels(model=model).observe(ttft)
    TPOT_HISTOGRAM.labels(model=model).observe(tpot)
    TPS_GAUGE.labels(model=model).set(tps)


def get_metrics_text() -> str:
    """Get Prometheus metrics as text."""
    return generate_latest().decode("utf-8")
