"""Synthetic data generator for testing and demos."""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import DATA_DIR


# ============================================================
# Synthetic Jira tickets
# ============================================================

CUSTOMERS = ["ExampleCorp", "AcmeAI", "DataFlowInc", "CloudPeak", "NeuralOps"]
MODELS = ["Llama-3-8B", "Mistral-7B", "Gemma-2-9B", "Phi-3-medium", "Qwen2-7B"]
RUNTIMES = ["vLLM", "TGI", "llama.cpp", "TensorRT-LLM", "ONNX Runtime"]
ISSUE_TYPES = ["Bug", "Performance", "Regression", "Configuration", "Capacity"]

SYNTHETIC_TICKETS = [
    {
        "key": "BUG-101",
        "summary": "TTFT increased from 800ms to 2.4s after upgrading vLLM runtime",
        "customer": "ExampleCorp",
        "model": "Llama-3-8B",
        "runtime": "vLLM",
        "issue_type": "Performance",
        "status": "Open",
        "input_tokens": 4096,
        "output_tokens": 256,
        "batch_size": 8,
        "concurrency": 16,
        "gpu_utilization_percent": 47,
        "memory_utilization_percent": 82,
        "ttft_observed_ms": 2400,
        "ttft_expected_ms": 1000,
        "tps_observed": 12.5,
        "tps_expected": 30.0,
        "error_count": 0,
        "description": "Customer reports TTFT increased from 800ms to 2.4 seconds after upgrading runtime from vLLM 0.2.x to 0.3.x. GPU utilization dropped from 72% to 47%. Memory utilization increased from 65% to 82%.",
        "expected_labels": ["performance", "regression", "ttft"],
    },
    {
        "key": "BUG-102",
        "summary": "OOM error when loading Mistral-7B with batch_size=32",
        "customer": "AcmeAI",
        "model": "Mistral-7B",
        "runtime": "vLLM",
        "issue_type": "Bug",
        "status": "Open",
        "input_tokens": 2048,
        "output_tokens": 512,
        "batch_size": 32,
        "concurrency": 1,
        "gpu_utilization_percent": 95,
        "memory_utilization_percent": 99,
        "ttft_observed_ms": 350,
        "ttft_expected_ms": 300,
        "tps_observed": 0,
        "tps_expected": 25.0,
        "error_count": 15,
        "description": "CUDA OOM error when batch_size exceeds 24 on A100-40GB. Works fine with batch_size=16.",
        "expected_labels": ["oom", "memory", "batch-size"],
    },
    {
        "key": "BUG-103",
        "summary": "TPS degraded 40% after quantization to INT4",
        "customer": "DataFlowInc",
        "model": "Llama-3-8B",
        "runtime": "llama.cpp",
        "issue_type": "Performance",
        "status": "Resolved",
        "input_tokens": 1024,
        "output_tokens": 128,
        "batch_size": 1,
        "concurrency": 1,
        "gpu_utilization_percent": 62,
        "memory_utilization_percent": 45,
        "ttft_observed_ms": 450,
        "ttft_expected_ms": 200,
        "tps_observed": 15.0,
        "tps_expected": 25.0,
        "error_count": 0,
        "description": "After INT4 quantization, TPS dropped from 25 to 15. TTFT doubled. Memory usage reduced as expected.",
        "expected_labels": ["quantization", "performance", "tps"],
    },
    {
        "key": "BUG-104",
        "summary": "Model returns empty responses under concurrency > 8",
        "customer": "CloudPeak",
        "model": "Gemma-2-9B",
        "runtime": "TGI",
        "issue_type": "Bug",
        "status": "Open",
        "input_tokens": 512,
        "output_tokens": 256,
        "batch_size": 4,
        "concurrency": 16,
        "gpu_utilization_percent": 38,
        "memory_utilization_percent": 55,
        "ttft_observed_ms": 100,
        "ttft_expected_ms": 200,
        "tps_observed": 0,
        "tps_expected": 20.0,
        "error_count": 42,
        "description": "When concurrency exceeds 8, model returns empty strings. No error in logs. Works fine at concurrency <= 8.",
        "expected_labels": ["concurrency", "empty-response", "stability"],
    },
    {
        "key": "BUG-105",
        "summary": "KV cache exhaustion after 2 hours of continuous inference",
        "customer": "NeuralOps",
        "model": "Phi-3-medium",
        "runtime": "vLLM",
        "issue_type": "Performance",
        "status": "Open",
        "input_tokens": 8192,
        "output_tokens": 1024,
        "batch_size": 4,
        "concurrency": 4,
        "gpu_utilization_percent": 88,
        "memory_utilization_percent": 94,
        "ttft_observed_ms": 5000,
        "ttft_expected_ms": 1500,
        "tps_observed": 5.0,
        "tps_expected": 15.0,
        "error_count": 8,
        "description": "After ~2 hours of continuous operation, KV cache fills up causing OOM. Requires service restart. TTFT degrades linearly over time.",
        "expected_labels": ["kv-cache", "memory-leak", "stability"],
    },
    {
        "key": "BUG-106",
        "summary": "GPU not detected - falling back to CPU inference",
        "customer": "ExampleCorp",
        "model": "Qwen2-7B",
        "runtime": "vLLM",
        "issue_type": "Configuration",
        "status": "Resolved",
        "input_tokens": 1024,
        "output_tokens": 256,
        "batch_size": 1,
        "concurrency": 1,
        "gpu_utilization_percent": 0,
        "memory_utilization_percent": 78,
        "ttft_observed_ms": 15000,
        "ttft_expected_ms": 500,
        "tps_observed": 1.2,
        "tps_expected": 20.0,
        "error_count": 0,
        "description": "vLLM not detecting Intel GPU. Running on CPU fallback. 100x slower than expected.",
        "expected_labels": ["gpu-detection", "configuration", "intel-gpu"],
    },
    {
        "key": "BUG-107",
        "summary": "Token generation stalls at exactly 128 tokens",
        "customer": "AcmeAI",
        "model": "Mistral-7B",
        "runtime": "TGI",
        "issue_type": "Bug",
        "status": "Open",
        "input_tokens": 2048,
        "output_tokens": 256,
        "batch_size": 1,
        "concurrency": 1,
        "gpu_utilization_percent": 55,
        "memory_utilization_percent": 60,
        "ttft_observed_ms": 300,
        "ttft_expected_ms": 300,
        "tps_observed": 22.0,
        "tps_expected": 22.0,
        "error_count": 5,
        "description": "Generation consistently stops at 128 tokens despite max_new_tokens=256. EOS token not being generated - appears to be a stop condition bug.",
        "expected_labels": ["generation-stall", "max-tokens", "stop-condition"],
    },
    {
        "key": "BUG-108",
        "summary": "TTFT spikes to 8s during model weight loading phase",
        "customer": "DataFlowInc",
        "model": "Llama-3-8B",
        "runtime": "TensorRT-LLM",
        "issue_type": "Performance",
        "status": "Open",
        "input_tokens": 4096,
        "output_tokens": 512,
        "batch_size": 1,
        "concurrency": 1,
        "gpu_utilization_percent": 35,
        "memory_utilization_percent": 40,
        "ttft_observed_ms": 8000,
        "ttft_expected_ms": 500,
        "tps_observed": 35.0,
        "tps_expected": 35.0,
        "error_count": 0,
        "description": "First request after cold start takes 8s TTFT due to engine compilation. Subsequent requests are fine at 500ms.",
        "expected_labels": ["cold-start", "ttft", "tensorrt"],
    },
    {
        "key": "BUG-109",
        "summary": "Throughput drops 60% with prompt_length > 4096",
        "customer": "CloudPeak",
        "model": "Llama-3-8B",
        "runtime": "vLLM",
        "issue_type": "Performance",
        "status": "Open",
        "input_tokens": 8192,
        "output_tokens": 256,
        "batch_size": 4,
        "concurrency": 8,
        "gpu_utilization_percent": 70,
        "memory_utilization_percent": 75,
        "ttft_observed_ms": 3200,
        "ttft_expected_ms": 1000,
        "tps_observed": 10.0,
        "tps_expected": 25.0,
        "error_count": 0,
        "description": "With long prompts (>4K tokens), both TTFT and throughput degrade significantly. Prefill becomes the bottleneck.",
        "expected_labels": ["long-context", "prefill", "throughput"],
    },
    {
        "key": "BUG-110",
        "summary": "CUDA kernel launch failures on Intel Arc A770",
        "customer": "NeuralOps",
        "model": "Mistral-7B",
        "runtime": "llama.cpp",
        "issue_type": "Bug",
        "status": "Open",
        "input_tokens": 1024,
        "output_tokens": 128,
        "batch_size": 1,
        "concurrency": 1,
        "gpu_utilization_percent": 10,
        "memory_utilization_percent": 30,
        "ttft_observed_ms": 0,
        "ttft_expected_ms": 200,
        "tps_observed": 0,
        "tps_expected": 15.0,
        "error_count": 30,
        "description": "Intel Arc A770 GPU not fully supported. SYCL backend crashes on kernel launch. CPU fallback works but is 50x slower.",
        "expected_labels": ["intel-gpu", "sycl", "compatibility"],
    },
]


def generate_synthetic_tickets(count: int = 10) -> list[dict]:
    """Generate synthetic Jira tickets."""
    tickets = []
    base_date = datetime.now(timezone.utc)

    for i, tmpl in enumerate(SYNTHETIC_TICKETS[:count]):
        ticket = tmpl.copy()
        ticket["created"] = (base_date - timedelta(days=random.randint(1, 60))).isoformat()
        ticket["updated"] = (base_date - timedelta(days=random.randint(0, 5))).isoformat()
        tickets.append(ticket)

    return tickets


def save_synthetic_data():
    """Save synthetic tickets to data directory."""
    tickets = generate_synthetic_tickets()
    out_path = DATA_DIR / "synthetic_tickets.json"
    with open(out_path, "w") as f:
        json.dump(tickets, f, indent=2)
    print(f"Saved {len(tickets)} synthetic tickets to {out_path}")
    return tickets


if __name__ == "__main__":
    save_synthetic_data()
