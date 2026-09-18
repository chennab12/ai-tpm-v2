"""Central configuration for AI-TPM-v2."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-driven settings."""

    HF_TOKEN: str = ""
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    JIRA_BASE_URL: str = ""
    JIRA_API_TOKEN: str = ""
    JIRA_EMAIL: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
MLRUNS_DIR = BASE_DIR / "mlruns"

LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)


@dataclass
class BenchmarkConfig:
    """Benchmark run configuration."""

    model_name: str = "distilgpt2"
    max_new_tokens: int = 128
    num_runs: int = 5
    warmup_runs: int = 1
    input_token_count: int = 256
    batch_size: int = 1
    concurrency: int = 1
    prompt: str = (
        "Explain why AI inference latency and throughput matter "
        "to a technical program manager in simple terms."
    )
    csv_log_file: str = str(LOGS_DIR / "inference_results.csv")
    scorecard_file: str = str(LOGS_DIR / "scorecard.json")


@dataclass
class AcceptanceCriteria:
    """TPM acceptance thresholds for PASS/WARN/FAIL."""

    target_ttft_sec: float = 1.0
    target_p95_latency_sec: float = 10.0
    target_tps: float = 5.0
    target_tpot_ms: float = 200.0
    target_error_rate_percent: float = 1.0
    target_cpu_percent: float = 90.0
    target_ram_percent: float = 85.0
    warn_multiplier_ttft: float = 1.5
    warn_multiplier_latency: float = 1.25
    warn_factor_tps: float = 0.80
    warn_factor_tpot: float = 1.25


@dataclass
class GPUReadinessWeights:
    """Weights for GPU readiness scoring from Jira tickets."""

    functional_stability: float = 0.25
    performance_latency: float = 0.20
    performance_throughput: float = 0.15
    reliability: float = 0.15
    resource_utilization: float = 0.15
    regression_free: float = 0.10


DEFAULT_CRITERIA = AcceptanceCriteria()
DEFAULT_WEIGHTS = GPUReadinessWeights()
