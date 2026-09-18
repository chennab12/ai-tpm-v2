"""
V9 - Automatic Validation.

After applying a fix to a ticket, validate whether the fix resolved
the issue by re-running benchmarks and comparing against the ticket's
expected SLA values (TTFT, TPS, etc).
"""

from dataclasses import dataclass

from src.benchmark.engine import BenchmarkSummary, run_benchmark
from src.config import BenchmarkConfig


@dataclass
class ValidationResult:
    """Result of validating a fix against ticket expectations."""
    ticket_key: str
    validation_passed: bool
    status_per_metric: dict[str, str]  # metric -> PASS/WARN/FAIL
    observed: dict
    expected: dict
    delta: dict
    summary: str


def validate_fix(
    ticket: dict,
    model: str,
    max_new_tokens: int = 256,
    num_runs: int = 3,
) -> ValidationResult:
    """Run benchmark and compare against ticket expected values."""
    # Reproduce ticket's workload using synthetic prompt sized to input tokens
    input_tok = ticket.get("input_tokens", 1024)
    prompt = ("Benchmark prompt placeholder. " * max(1, input_tok // 20))

    if ticket.get("runtime", "").lower() == "vllm":
        model_or_alias = model
    else:
        model_or_alias = model

    summary: BenchmarkSummary = run_benchmark(BenchmarkConfig(
        model_name=model_or_alias,
        max_new_tokens=min(ticket.get("output_tokens", 256), max_new_tokens),
        num_runs=num_runs,
        warmup_runs=1,
        prompt=prompt,
    ))

    expected_ttft_ms = ticket.get("ttft_expected_ms", 1000)
    expected_tps = ticket.get("tps_expected", 20.0)
    observed_ttft_ms = summary.avg_ttft_sec * 1000
    observed_tps = summary.avg_tps
    observed_gpu = summary.avg_gpu_utilization
    expected_gpu = ticket.get("gpu_utilization_percent", 50)

    checks: dict[str, str] = {}
    # TTFT
    if observed_ttft_ms <= expected_ttft_ms * 1.2:
        checks["ttft"] = "PASS"
    elif observed_ttft_ms <= expected_ttft_ms * 1.5:
        checks["ttft"] = "WARN"
    else:
        checks["ttft"] = "FAIL"

    # TPS
    if observed_tps >= expected_tps * 0.8:
        checks["tps"] = "PASS"
    elif observed_tps >= expected_tps * 0.6:
        checks["tps"] = "WARN"
    else:
        checks["tps"] = "FAIL"

    # GPU utilization sanity (not saturated nor idle)
    if observed_gpu >= 20 and observed_gpu <= 95:
        checks["gpu_utilization"] = "PASS"
    elif observed_gpu == 0 and expected_gpu > 0:
        checks["gpu_utilization"] = "FAIL"
    else:
        checks["gpu_utilization"] = "WARN"

    passed = all(v == "PASS" for v in checks.values())

    return ValidationResult(
        ticket_key=ticket.get("key", "UNKNOWN"),
        validation_passed=passed,
        status_per_metric=checks,
        observed={
            "ttft_ms": round(observed_ttft_ms, 1),
            "tps": round(observed_tps, 1),
            "gpu_utilization": round(observed_gpu, 1),
            "p95_latency_sec": summary.p95_latency_sec,
        },
        expected={
            "ttft_ms": expected_ttft_ms,
            "tps": expected_tps,
            "gpu_utilization": expected_gpu,
        },
        delta={
            "ttft_delta_pct": round((observed_ttft_ms - expected_ttft_ms) / max(expected_ttft_ms, 1) * 100, 1),
            "tps_delta_pct": round((observed_tps - expected_tps) / max(expected_tps, 1) * 100, 1),
        },
        summary=(
            "FIX VALIDATED: criteria met."
            if passed
            else "RECOMMENDATION: re-check fix, criteria not fully met."
        ),
    )


def print_validation(result: ValidationResult):
    """Print validation summary."""
    print(f"\n{'='*65}")
    print(f"AUTOMATIC VALIDATION  |  Ticket: {result.ticket_key}")
    print(f"{'='*65}")
    for metric, status in result.status_per_metric.items():
        print(f"  {status:4s}  {metric}")
    print()
    print(f"  Observed:  TTFT={result.observed['ttft_ms']}ms  "
          f"TPS={result.observed['tps']}  GPU={result.observed['gpu_utilization']}%")
    print(f"  Expected:  TTFT={result.expected['ttft_ms']}ms  "
          f"TPS={result.expected['tps']}  GPU={result.expected['gpu_utilization']}%")
    print(f"  Delta:     TTFT={result.delta['ttft_delta_pct']}%  "
          f"TPS={result.delta['tps_delta_pct']}%")
    print()
    print(f"  >>> {result.summary} <<<")
    print()