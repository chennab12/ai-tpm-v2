"""
Core benchmark engine.

Handles model loading, inference, streaming TTFT/TPOT measurement,
system resource monitoring, and multi-run aggregation with P50/P95.
"""

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Thread
from typing import Any

import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from src.config import (
    BenchmarkConfig,
    DEFAULT_CRITERIA,
    LOGS_DIR,
)


# ============================================================
# Data classes
# ============================================================


@dataclass
class InferenceResult:
    """Single inference run result."""

    run: int
    timestamp: str
    success: bool
    error: str
    input_tokens: int
    output_tokens: int
    latency_sec: float
    ttft_sec: float
    tpot_sec: float
    tokens_per_sec: float
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    gpu_utilization: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    output_text: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results across all runs."""

    model_name: str
    device: str
    num_runs: int
    successful_runs: int
    failed_runs: int
    input_tokens: int
    max_new_tokens: int
    avg_latency_sec: float
    p50_latency_sec: float
    p95_latency_sec: float
    min_latency_sec: float
    max_latency_sec: float
    std_latency_sec: float
    avg_ttft_sec: float
    p50_ttft_sec: float
    p95_ttft_sec: float
    avg_tpot_sec: float
    avg_tps: float
    p50_tps: float
    p95_tps: float
    error_rate_percent: float
    avg_cpu_percent: float
    avg_ram_percent: float
    avg_gpu_utilization: float
    avg_gpu_memory_used_gb: float
    warmup_latency_sec: float
    results: list[InferenceResult] = field(default_factory=list)
    scorecard: dict = field(default_factory=dict)
    timestamp: str = ""


# ============================================================
# Helpers
# ============================================================


def _percentile(values: list[float], pct: float) -> float:
    """Calculate percentile from a list of values."""
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _gpu_metrics() -> dict[str, float]:
    """Read GPU metrics via PyTorch CUDA if available."""
    if not torch.cuda.is_available():
        return {
            "gpu_utilization": 0.0,
            "gpu_memory_used_gb": 0.0,
            "gpu_memory_total_gb": 0.0,
        }
    total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    used = torch.cuda.memory_allocated(0) / (1024**3)
    util = (used / total * 100) if total > 0 else 0.0
    return {
        "gpu_utilization": util,
        "gpu_memory_used_gb": round(used, 3),
        "gpu_memory_total_gb": round(total, 3),
    }


def _system_metrics() -> dict[str, float]:
    """Read CPU and RAM metrics."""
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": cpu,
        "ram_used_gb": round(mem.used / (1024**3), 3),
        "ram_percent": mem.percent,
    }


# ============================================================
# Model loader
# ============================================================


class ModelLoader:
    """Load and manage model + tokenizer."""

    def __init__(self, model_name: str, hf_token: str = ""):
        self.model_name = model_name
        self.hf_token = hf_token
        self.tokenizer = None
        self.model = None
        self.device = None

    def load(self):
        """Load tokenizer and model."""
        from src.config import settings

        token = self.hf_token or settings.HF_TOKEN or None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        return self

    def tokenize(self, text: str) -> dict[str, Any]:
        """Tokenize text and move to device."""
        enc = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}


# ============================================================
# Single inference runner
# ============================================================


def run_single_inference(
    loader: ModelLoader,
    inputs: dict,
    run_number: int,
    max_new_tokens: int = 128,
) -> InferenceResult:
    """Execute one streaming inference and capture all metrics."""
    streamer = TextIteratorStreamer(
        loader.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": loader.tokenizer.eos_token_id,
    }

    output_parts: list[str] = []
    first_token_time: float | None = None
    start = time.perf_counter()

    thread = Thread(target=loader.model.generate, kwargs=gen_kwargs)
    thread.start()

    try:
        for chunk in streamer:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            output_parts.append(chunk)
        thread.join()
        end = time.perf_counter()
        success, error = True, ""
    except Exception as exc:
        end = time.perf_counter()
        success, error = False, str(exc)

    output_text = "".join(output_parts)
    out_ids = loader.tokenizer(output_text, return_tensors="pt")["input_ids"]
    output_tokens = out_ids.shape[1]

    total_latency = end - start
    ttft = (first_token_time - start) if first_token_time else total_latency
    decode_duration = max(total_latency - ttft, 1e-6)
    tps = output_tokens / total_latency if total_latency > 0 else 0.0
    tpot = decode_duration / (output_tokens - 1) if output_tokens > 1 else decode_duration

    gpu = _gpu_metrics()
    sys_m = _system_metrics()

    return InferenceResult(
        run=run_number,
        timestamp=datetime.now(timezone.utc).isoformat(),
        success=success,
        error=error,
        input_tokens=inputs["input_ids"].shape[1],
        output_tokens=output_tokens,
        latency_sec=round(total_latency, 4),
        ttft_sec=round(ttft, 4),
        tpot_sec=round(tpot, 6),
        tokens_per_sec=round(tps, 2),
        cpu_percent=round(sys_m["cpu_percent"], 1),
        ram_used_gb=sys_m["ram_used_gb"],
        ram_percent=round(sys_m["ram_percent"], 1),
        gpu_utilization=round(gpu["gpu_utilization"], 1),
        gpu_memory_used_gb=gpu["gpu_memory_used_gb"],
        gpu_memory_total_gb=gpu["gpu_memory_total_gb"],
        output_text=output_text,
    )


# ============================================================
# CSV logger
# ============================================================

CSV_FIELDS = [
    "timestamp", "run", "success", "error", "input_tokens",
    "output_tokens", "latency_sec", "ttft_sec", "tpot_sec",
    "tokens_per_sec", "cpu_percent", "ram_used_gb", "ram_percent",
    "gpu_utilization", "gpu_memory_used_gb", "gpu_memory_total_gb",
]


def append_csv(results: list[InferenceResult], path: str):
    """Append results to CSV log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in CSV_FIELDS}
            writer.writerow(row)


# ============================================================
# Scorecard generator
# ============================================================


def generate_scorecard(
    summary: "BenchmarkSummary",
    criteria=None,
) -> dict:
    """Generate PASS/WARN/FAIL scorecard."""
    if criteria is None:
        from src.config import DEFAULT_CRITERIA as criteria

    sc: dict[str, Any] = {}

    # TTFT check
    if summary.avg_ttft_sec <= criteria.target_ttft_sec:
        sc["ttft"] = {"status": "PASS", "value": summary.avg_ttft_sec, "target": f"<={criteria.target_ttft_sec}s"}
    elif summary.avg_ttft_sec <= criteria.target_ttft_sec * criteria.warn_multiplier_ttft:
        sc["ttft"] = {"status": "WARN", "value": summary.avg_ttft_sec, "target": f"<={criteria.target_ttft_sec}s"}
    else:
        sc["ttft"] = {"status": "FAIL", "value": summary.avg_ttft_sec, "target": f"<={criteria.target_ttft_sec}s"}

    # P95 latency
    if summary.p95_latency_sec <= criteria.target_p95_latency_sec:
        sc["p95_latency"] = {"status": "PASS", "value": summary.p95_latency_sec, "target": f"<={criteria.target_p95_latency_sec}s"}
    elif summary.p95_latency_sec <= criteria.target_p95_latency_sec * criteria.warn_multiplier_latency:
        sc["p95_latency"] = {"status": "WARN", "value": summary.p95_latency_sec, "target": f"<={criteria.target_p95_latency_sec}s"}
    else:
        sc["p95_latency"] = {"status": "FAIL", "value": summary.p95_latency_sec, "target": f"<={criteria.target_p95_latency_sec}s"}

    # TPS
    if summary.avg_tps >= criteria.target_tps:
        sc["tps"] = {"status": "PASS", "value": summary.avg_tps, "target": f">={criteria.target_tps}"}
    elif summary.avg_tps >= criteria.target_tps * criteria.warn_factor_tps:
        sc["tps"] = {"status": "WARN", "value": summary.avg_tps, "target": f">={criteria.target_tps}"}
    else:
        sc["tps"] = {"status": "FAIL", "value": summary.avg_tps, "target": f">={criteria.target_tps}"}

    # TPOT
    tpot_ms = summary.avg_tpot_sec * 1000
    if tpot_ms <= criteria.target_tpot_ms:
        sc["tpot"] = {"status": "PASS", "value": round(tpot_ms, 2), "target": f"<={criteria.target_tpot_ms}ms"}
    elif tpot_ms <= criteria.target_tpot_ms * criteria.warn_factor_tpot:
        sc["tpot"] = {"status": "WARN", "value": round(tpot_ms, 2), "target": f"<={criteria.target_tpot_ms}ms"}
    else:
        sc["tpot"] = {"status": "FAIL", "value": round(tpot_ms, 2), "target": f"<={criteria.target_tpot_ms}ms"}

    # Error rate
    if summary.error_rate_percent <= criteria.target_error_rate_percent:
        sc["error_rate"] = {"status": "PASS", "value": summary.error_rate_percent, "target": f"<={criteria.target_error_rate_percent}%"}
    elif summary.error_rate_percent <= 5.0:
        sc["error_rate"] = {"status": "WARN", "value": summary.error_rate_percent, "target": f"<={criteria.target_error_rate_percent}%"}
    else:
        sc["error_rate"] = {"status": "FAIL", "value": summary.error_rate_percent, "target": f"<={criteria.target_error_rate_percent}%"}

    # CPU
    if summary.avg_cpu_percent <= criteria.target_cpu_percent:
        sc["cpu"] = {"status": "PASS", "value": summary.avg_cpu_percent, "target": f"<={criteria.target_cpu_percent}%"}
    else:
        sc["cpu"] = {"status": "FAIL" if summary.avg_cpu_percent > 95 else "WARN", "value": summary.avg_cpu_percent, "target": f"<={criteria.target_cpu_percent}%"}

    # RAM
    if summary.avg_ram_percent <= criteria.target_ram_percent:
        sc["ram"] = {"status": "PASS", "value": summary.avg_ram_percent, "target": f"<={criteria.target_ram_percent}%"}
    else:
        sc["ram"] = {"status": "FAIL" if summary.avg_ram_percent > 95 else "WARN", "value": summary.avg_ram_percent, "target": f"<={criteria.target_ram_percent}%"}

    # Overall
    statuses = [v["status"] for v in sc.values()]
    if "FAIL" in statuses:
        sc["overall"] = "FAIL"
    elif "WARN" in statuses:
        sc["overall"] = "WARN"
    else:
        sc["overall"] = "PASS"

    return sc


# ============================================================
# Full benchmark runner
# ============================================================


def run_benchmark(config: BenchmarkConfig | None = None) -> BenchmarkSummary:
    """Execute a full benchmark session."""
    if config is None:
        config = BenchmarkConfig()

    print(f"\n{'='*65}")
    print(f"AI-TPM BENCHMARK  |  Model: {config.model_name}")
    print(f"{'='*65}")

    # Load model
    print("\n[1/4] Loading model...")
    loader = ModelLoader(config.model_name).load()
    print(f"  Model:  {config.model_name}")
    print(f"  Device: {loader.device}")

    # Tokenize
    print("\n[2/4] Tokenizing prompt...")
    inputs = loader.tokenize(config.prompt)
    input_tok = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {input_tok}")

    # Warm-up
    print(f"\n[3/4] Warm-up ({config.warmup_runs} run(s))...")
    warmup_latency = 0.0
    for i in range(config.warmup_runs):
        wr = run_single_inference(loader, inputs, 0, config.max_new_tokens)
        warmup_latency += wr.latency_sec
        print(f"  Warm-up {i+1}: {wr.latency_sec:.2f}s")
    warmup_latency /= max(config.warmup_runs, 1)

    # Benchmark runs
    print(f"\n[4/4] Benchmark runs ({config.num_runs})...")
    results: list[InferenceResult] = []
    for n in range(1, config.num_runs + 1):
        r = run_single_inference(loader, inputs, n, config.max_new_tokens)
        results.append(r)
        status = "OK" if r.success else "FAIL"
        print(
            f"  Run {n}/{config.num_runs} [{status}]  "
            f"lat={r.latency_sec:.2f}s  ttft={r.ttft_sec:.2f}s  "
            f"tpot={r.tpot_sec*1000:.1f}ms  tps={r.tokens_per_sec:.1f}  "
            f"gpu={r.gpu_utilization:.0f}%  cpu={r.cpu_percent:.0f}%"
        )

    # Aggregate
    ok = [r for r in results if r.success]
    latencies = [r.latency_sec for r in ok]
    ttfts = [r.ttft_sec for r in ok]
    tpots = [r.tpot_sec for r in ok]
    tps_vals = [r.tokens_per_sec for r in ok]

    summary = BenchmarkSummary(
        model_name=config.model_name,
        device=str(loader.device),
        num_runs=config.num_runs,
        successful_runs=len(ok),
        failed_runs=len(results) - len(ok),
        input_tokens=input_tok,
        max_new_tokens=config.max_new_tokens,
        avg_latency_sec=round(statistics.mean(latencies), 4) if latencies else 0,
        p50_latency_sec=round(_percentile(latencies, 0.50), 4) if latencies else 0,
        p95_latency_sec=round(_percentile(latencies, 0.95), 4) if latencies else 0,
        min_latency_sec=round(min(latencies), 4) if latencies else 0,
        max_latency_sec=round(max(latencies), 4) if latencies else 0,
        std_latency_sec=round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0,
        avg_ttft_sec=round(statistics.mean(ttfts), 4) if ttfts else 0,
        p50_ttft_sec=round(_percentile(ttfts, 0.50), 4) if ttfts else 0,
        p95_ttft_sec=round(_percentile(ttfts, 0.95), 4) if ttfts else 0,
        avg_tpot_sec=round(statistics.mean(tpots), 6) if tpots else 0,
        avg_tps=round(statistics.mean(tps_vals), 2) if tps_vals else 0,
        p50_tps=round(_percentile(tps_vals, 0.50), 2) if tps_vals else 0,
        p95_tps=round(_percentile(tps_vals, 0.95), 2) if tps_vals else 0,
        error_rate_percent=round((len(results) - len(ok)) / max(len(results), 1) * 100, 1),
        avg_cpu_percent=round(statistics.mean([r.cpu_percent for r in ok]), 1) if ok else 0,
        avg_ram_percent=round(statistics.mean([r.ram_percent for r in ok]), 1) if ok else 0,
        avg_gpu_utilization=round(statistics.mean([r.gpu_utilization for r in ok]), 1) if ok else 0,
        avg_gpu_memory_used_gb=round(statistics.mean([r.gpu_memory_used_gb for r in ok]), 3) if ok else 0,
        warmup_latency_sec=round(warmup_latency, 4),
        results=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Scorecard
    summary.scorecard = generate_scorecard(summary)

    # CSV log
    append_csv(results, config.csv_log_file)

    # Scorecard file
    os.makedirs(os.path.dirname(config.scorecard_file), exist_ok=True)
    sc_out = {
        "timestamp": summary.timestamp,
        "model": summary.model_name,
        "device": summary.device,
        "scorecard": summary.scorecard,
        "summary": {
            k: getattr(summary, k)
            for k in [
                "avg_latency_sec", "p50_latency_sec", "p95_latency_sec",
                "avg_ttft_sec", "avg_tpot_sec", "avg_tps",
                "error_rate_percent", "avg_cpu_percent", "avg_ram_percent",
                "avg_gpu_utilization",
            ]
        },
    }
    with open(config.scorecard_file, "w") as f:
        json.dump(sc_out, f, indent=2, default=str)

    return summary


def print_summary(summary: BenchmarkSummary):
    """Pretty-print benchmark summary."""
    print(f"\n{'='*65}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*65}")
    print(f"Model:               {summary.model_name}")
    print(f"Device:              {summary.device}")
    print(f"Runs:                {summary.successful_runs}/{summary.num_runs} OK")
    print(f"Input tokens:        {summary.input_tokens}")
    print(f"Max output tokens:   {summary.max_new_tokens}")
    print()
    print(f"Avg latency:         {summary.avg_latency_sec:.2f}s")
    print(f"P50 latency:         {summary.p50_latency_sec:.2f}s")
    print(f"P95 latency:         {summary.p95_latency_sec:.2f}s")
    print(f"Min/Max:             {summary.min_latency_sec:.2f}s / {summary.max_latency_sec:.2f}s")
    print(f"Std dev:             {summary.std_latency_sec:.4f}s")
    print()
    print(f"Avg TTFT:            {summary.avg_ttft_sec:.2f}s")
    print(f"P50 TTFT:            {summary.p50_ttft_sec:.2f}s")
    print(f"P95 TTFT:            {summary.p95_ttft_sec:.2f}s")
    print()
    print(f"Avg TPOT:            {summary.avg_tpot_sec*1000:.2f} ms/token")
    print(f"Avg TPS:             {summary.avg_tps:.2f} tok/s")
    print(f"P50 TPS:             {summary.p50_tps:.2f} tok/s")
    print(f"P95 TPS:             {summary.p95_tps:.2f} tok/s")
    print()
    print(f"Error rate:          {summary.error_rate_percent:.1f}%")
    print(f"Avg CPU:             {summary.avg_cpu_percent:.1f}%")
    print(f"Avg RAM:             {summary.avg_ram_percent:.1f}%")
    print(f"Avg GPU util:        {summary.avg_gpu_utilization:.1f}%")
    print(f"Warmup latency:      {summary.warmup_latency_sec:.2f}s")
    print()

    # Scorecard
    print(f"{'='*65}")
    print("TPM SCORECARD")
    print(f"{'='*65}")
    for key, val in summary.scorecard.items():
        if key == "overall":
            continue
        icon = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[val["status"]]
        print(f"  {icon}  {key:15s}  {val['value']}  (target: {val['target']})")
    print()
    print(f"  >>> OVERALL: {summary.scorecard.get('overall', 'N/A')} <<<")
    print()
