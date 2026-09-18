"""MLflow experiment tracking integration (with file-based fallback)."""

import json
import os
from datetime import datetime, timezone

from src.config import LOGS_DIR, settings

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def init_mlflow(model_name: str = "ai-tpm"):
    """Initialize MLflow tracking."""
    if not _HAS_MLFLOW:
        return
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"ai-tpm-{model_name}")


def _fallback_log(summary, direction: str = "created") -> str:
    """Write benchmark results to a local JSON file when MLflow is unavailable."""
    fallback_dir = LOGS_DIR / "experiments"
    os.makedirs(fallback_dir, exist_ok=True)
    run_id = f"local-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    entry = {
        "run_id": run_id,
        "direction": direction,
        "timestamp": summary.timestamp,
        "model": summary.model_name,
        "device": summary.device,
        "metrics": {
            "avg_latency_sec": summary.avg_latency_sec,
            "p50_latency_sec": summary.p50_latency_sec,
            "p95_latency_sec": summary.p95_latency_sec,
            "avg_ttft_sec": summary.avg_ttft_sec,
            "avg_tpot_sec": summary.avg_tpot_sec,
            "avg_tps": summary.avg_tps,
            "error_rate_percent": summary.error_rate_percent,
            "avg_gpu_utilization": summary.avg_gpu_utilization,
            "avg_cpu_percent": summary.avg_cpu_percent,
        },
        "scorecard": summary.scorecard,
    }
    log_path = fallback_dir / f"{summary.model_name}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return run_id


def log_benchmark(summary) -> str:
    """Log a full benchmark run. Returns run_id."""
    if not _HAS_MLFLOW:
        return _fallback_log(summary)

    init_mlflow(summary.model_name)

    with mlflow.start_run(run_name=f"bench-{summary.model_name}-{datetime.now(timezone.utc).strftime('%H%M%S')}"):
        mlflow.set_tag("model", summary.model_name)
        mlflow.set_tag("device", summary.device)

        mlflow.log_params({
            "model_name": summary.model_name,
            "device": summary.device,
            "num_runs": summary.num_runs,
            "input_tokens": summary.input_tokens,
            "max_new_tokens": summary.max_new_tokens,
            "warmup_latency_sec": summary.warmup_latency_sec,
        })

        mlflow.log_metrics({
            "avg_latency_sec": summary.avg_latency_sec,
            "p50_latency_sec": summary.p50_latency_sec,
            "p95_latency_sec": summary.p95_latency_sec,
            "min_latency_sec": summary.min_latency_sec,
            "max_latency_sec": summary.max_latency_sec,
            "std_latency_sec": summary.std_latency_sec,
            "avg_ttft_sec": summary.avg_ttft_sec,
            "p50_ttft_sec": summary.p50_ttft_sec,
            "p95_ttft_sec": summary.p95_ttft_sec,
            "avg_tpot_sec": summary.avg_tpot_sec,
            "avg_tps": summary.avg_tps,
            "error_rate_percent": summary.error_rate_percent,
            "avg_cpu_percent": summary.avg_cpu_percent,
            "avg_ram_percent": summary.avg_ram_percent,
            "avg_gpu_utilization": summary.avg_gpu_utilization,
        })

        mlflow.log_dict(
            {"timestamp": summary.timestamp, "scorecard": summary.scorecard},
            "scorecard.json",
        )

        csv_path = LOGS_DIR / "inference_results.csv"
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path))

        run_id = mlflow.active_run().info.run_id

    return run_id


def log_metrics_batch(metrics: dict, tags: dict = None, run_name: str = "predictions"):
    """Log a batch of metrics to MLflow (e.g. per-ticket predictions)."""
    if not _HAS_MLFLOW:
        return _fallback_log_metrics(metrics, tags, run_name)

    init_mlflow("ai-tpm")
    with mlflow.start_run(run_name=run_name):
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(str(k), str(v))
        for k, v in metrics.items():
            mlflow.log_metric(str(k), float(
                v if v is not None else 0.0
            ) if isinstance(v, (int, float)) else 0.0)
        return mlflow.active_run().info.run_id


def _fallback_log_metrics(metrics: dict, tags: dict = None, run_name: str = "predictions") -> str:
    fallback_dir = LOGS_DIR / "experiments"
    os.makedirs(fallback_dir, exist_ok=True)
    run_id = f"predict-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    entry = {"run_id": run_id, "run_name": run_name, "tags": tags or {}, "metrics": metrics}
    with open(fallback_dir / "predictions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return run_id


def is_available() -> bool:
    return _HAS_MLFLOW