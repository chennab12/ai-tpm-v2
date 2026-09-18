"""
Jira Ticket Analyzer - GPU Readiness Scorer.

Analyzes Jira tickets related to AI inference to produce:
- GPU model readiness percentage
- Readiness breakdown by category
- Improvement suggestions
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import DATA_DIR, DEFAULT_WEIGHTS, GPUReadinessWeights


# ============================================================
# Readiness dimensions
# ============================================================

DIMENSIONS = {
    "functional_stability": {
        "description": "Can the model run without errors?",
        "keywords": ["oom", "crash", "error", "empty-response", "stall", "exception", "timeout"],
        "penalty_per_ticket": 12,
    },
    "performance_latency": {
        "description": "Are latency targets (TTFT, TPOT) being met?",
        "keywords": ["ttft", "latency", "tpot", "cold-start", "prefill"],
        "penalty_per_ticket": 10,
    },
    "performance_throughput": {
        "description": "Is throughput (TPS) meeting targets?",
        "keywords": ["tps", "throughput", "batch", "concurrency"],
        "penalty_per_ticket": 8,
    },
    "reliability": {
        "description": "Is the system stable over time?",
        "keywords": ["memory-leak", "kv-cache", "stability", "degradation", "regression"],
        "penalty_per_ticket": 10,
    },
    "resource_utilization": {
        "description": "Are GPU/CPU/RAM resources used efficiently?",
        "keywords": ["gpu-detection", "cpu-fallback", "memory", "utilization", "gpu"],
        "penalty_per_ticket": 7,
    },
    "regression_free": {
        "description": "Are there regressions from previous versions?",
        "keywords": ["regression", "degraded", "worse", "increased"],
        "penalty_per_ticket": 9,
    },
}


@dataclass
class ReadinessResult:
    """GPU readiness analysis result."""
    model: str
    overall_readiness_pct: float
    dimension_scores: dict[str, float]
    dimension_details: dict[str, dict]
    tickets_analyzed: int
    open_bugs: int
    resolved_bugs: int
    severity_breakdown: dict[str, int]
    suggestions: list[str]
    readiness_level: str  # "Production Ready", "Near Ready", "Not Ready"


def _classify_ticket(ticket: dict) -> dict[str, float]:
    """Classify a ticket into readiness dimensions with severity."""
    text = f"{ticket.get('summary', '')} {ticket.get('description', '')}".lower()
    labels = [l.lower() for l in ticket.get("expected_labels", [])]

    scores: dict[str, float] = {}
    for dim, meta in DIMENSIONS.items():
        score = 0.0
        matched = False
        for kw in meta["keywords"]:
            if kw in text or kw in " ".join(labels):
                matched = True
                break
        if matched:
            # Severity multiplier based on ticket characteristics
            severity = 1.0
            if ticket.get("status") == "Open":
                severity = 1.5
            if ticket.get("error_count", 0) > 10:
                severity *= 1.3
            obs = ticket.get("ttft_observed_ms", 0)
            exp = ticket.get("ttft_expected_ms", 0)
            if exp > 0 and obs > exp * 2:
                severity *= 1.2
            score = meta["penalty_per_ticket"] * severity
        scores[dim] = score
    return scores


def analyze_tickets(
    tickets: list[dict],
    model_filter: str = "",
    weights: GPUReadinessWeights | None = None,
) -> ReadinessResult:
    """Analyze a list of Jira tickets and compute GPU readiness."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Filter by model if specified
    if model_filter:
        tickets = [t for t in tickets if t.get("model", "").lower() == model_filter.lower()]

    # Classify each ticket
    dim_totals: dict[str, float] = {d: 0.0 for d in DIMENSIONS}
    dim_ticket_counts: dict[str, int] = {d: 0 for d in DIMENSIONS}
    open_bugs = 0
    resolved_bugs = 0
    severity_counts: dict[str, int] = {"Open": 0, "Resolved": 0, "In Progress": 0}

    for ticket in tickets:
        classification = _classify_ticket(ticket)
        status = ticket.get("status", "Unknown")
        if status == "Open":
            open_bugs += 1
        elif status == "Resolved":
            resolved_bugs += 1
        severity_counts[status] = severity_counts.get(status, 0) + 1

        for dim, penalty in classification.items():
            if penalty > 0:
                dim_totals[dim] += penalty
                dim_ticket_counts[dim] += 1

    # Compute dimension scores (100 = no issues, lower = worse)
    dimension_scores: dict[str, float] = {}
    dimension_details: dict[str, dict] = {}
    for dim in DIMENSIONS:
        max_possible = len(tickets) * DIMENSIONS[dim]["penalty_per_ticket"] * 1.5
        actual = dim_totals[dim]
        if max_possible > 0:
            score = max(0, 100 - (actual / max_possible * 100))
        else:
            score = 100.0
        dimension_scores[dim] = round(score, 1)
        dimension_details[dim] = {
            "score": round(score, 1),
            "tickets_affected": dim_ticket_counts[dim],
            "total_penalty": round(dim_totals[dim], 1),
            "description": DIMENSIONS[dim]["description"],
        }

    # Weighted overall score
    weight_map = {
        "functional_stability": weights.functional_stability,
        "performance_latency": weights.performance_latency,
        "performance_throughput": weights.performance_throughput,
        "reliability": weights.reliability,
        "resource_utilization": weights.resource_utilization,
        "regression_free": weights.regression_free,
    }
    total_weight = sum(weight_map.values())

    if len(tickets) == 0:
        # No ticket evidence for this model -> unverified
        dimension_scores = {d: 25.0 for d in DIMENSIONS}
        dimension_details = {
            d: {
                "score": 25.0,
                "tickets_affected": 0,
                "total_penalty": 0.0,
                "description": DIMENSIONS[d]["description"],
            }
            for d in DIMENSIONS
        }
        overall = 25.0
        level = "Unverified - No Ticket Evidence"
        suggestions = [
            "No Jira tickets reference this model. GPU readiness cannot be certified without evidence.",
            "Run a full benchmark suite and record results (TTFT, TPOT, TPS, P95) as tickets.",
            "File at least one validation ticket per MLOps milestone to establish tracking.",
            "Consider deploying on Intel GPU and capturing compatibility (SYCL/oneAPI/IPEX) results.",
        ]
        model_name = model_filter or "Unknown"
        return ReadinessResult(
            model=model_name,
            overall_readiness_pct=25.0,
            dimension_scores=dimension_scores,
            dimension_details=dimension_details,
            tickets_analyzed=0,
            open_bugs=0,
            resolved_bugs=0,
            severity_breakdown={},
            suggestions=suggestions,
            readiness_level=level,
        )

    overall = sum(
        dimension_scores[d] * weight_map[d] for d in dimension_scores
    ) / total_weight

    # Readiness level
    if overall >= 80:
        level = "Production Ready"
    elif overall >= 60:
        level = "Near Ready"
    elif overall >= 40:
        level = "Needs Work"
    else:
        level = "Not Ready"

    # Generate suggestions
    suggestions = _generate_suggestions(dimension_scores, dim_ticket_counts, tickets)

    # Determine model name
    model_name = model_filter or (tickets[0].get("model", "Unknown") if tickets else "Unknown")

    return ReadinessResult(
        model=model_name,
        overall_readiness_pct=round(overall, 1),
        dimension_scores=dimension_scores,
        dimension_details=dimension_details,
        tickets_analyzed=len(tickets),
        open_bugs=open_bugs,
        resolved_bugs=resolved_bugs,
        severity_breakdown=severity_counts,
        suggestions=suggestions,
        readiness_level=level,
    )


def _generate_suggestions(
    scores: dict[str, float],
    ticket_counts: dict[str, int],
    tickets: list[dict],
) -> list[str]:
    """Generate actionable improvement suggestions."""
    suggestions = []

    if scores.get("functional_stability", 100) < 80:
        suggestions.append(
            "CRITICAL: Functional stability issues detected. Prioritize fixing OOM/crash bugs "
            "before any performance tuning. Run extended soak tests (24h+) to catch edge cases."
        )
    if scores.get("performance_latency", 100) < 70:
        suggestions.append(
            "TTFT is a key bottleneck. Consider: (1) enabling prefix caching, (2) reducing max_model_len, "
            "(3) using continuous batching, (4) profiling prefill vs decode phases separately."
        )
    if scores.get("performance_throughput", 100) < 70:
        suggestions.append(
            "Throughput below target. Try: (1) increasing batch_size, (2) enabling tensor parallelism, "
            "(3) using quantization (AWQ/GPTQ) to fit larger batches, (4) tuning max_num_batched_tokens."
        )
    if scores.get("reliability", 100) < 75:
        suggestions.append(
            "Stability issues over time detected. Implement: (1) KV cache budget limits, "
            "(2) automatic memory monitoring, (3) graceful degradation, (4) periodic restart strategy."
        )
    if scores.get("resource_utilization", 100) < 70:
        suggestions.append(
            "Resource utilization suboptimal. Check: (1) GPU driver compatibility, "
            "(2) CUDA/SYCL version matching, (3) memory allocation strategy, (4) CPU offloading config."
        )
    if scores.get("regression_free", 100) < 80:
        suggestions.append(
            "Regressions detected. Establish: (1) automated regression benchmark suite, "
            "(2) compare against known-good baseline, (3) pin runtime versions, (4) A/B test upgrades."
        )

    # Model-specific suggestions from ticket patterns
    text_blob = " ".join(
        f"{t.get('summary', '')} {t.get('description', '')}"
        for t in tickets
    ).lower()

    if "intel" in text_blob and ("arc" in text_blob or "gpu" in text_blob):
        suggestions.append(
            "Intel GPU compatibility: Ensure oneAPI toolkit is installed, "
            "verify SYCL backend support, test with ipex extension."
        )
    if "quantiz" in text_blob:
        suggestions.append(
            "Quantization trade-off detected. Benchmark FP16 vs INT8 vs INT4 "
            "to find optimal accuracy/speed balance for your use case."
        )
    if "concurrency" in text_blob:
        suggestions.append(
            "Concurrency issues found. Profile with: (1) increasing concurrency gradually, "
            "(2) monitoring request queue depth, (3) testing connection pooling."
        )

    if not suggestions:
        suggestions.append(
            "No critical issues detected. Continue monitoring with automated benchmarks."
        )

    return suggestions


def load_and_analyze(
    ticket_path: str | Path | None = None,
    model: str = "",
) -> ReadinessResult:
    """Load tickets from file and run analysis."""
    if ticket_path is None:
        ticket_path = DATA_DIR / "synthetic_tickets.json"
    else:
        ticket_path = Path(ticket_path)

    with open(ticket_path) as f:
        tickets = json.load(f)

    return analyze_tickets(tickets, model_filter=model)


def print_readiness(result: ReadinessResult):
    """Pretty-print readiness result."""
    print(f"\n{'='*65}")
    print(f"GPU READINESS REPORT  |  {result.model}")
    print(f"{'='*65}")
    print(f"Readiness Level:  {result.readiness_level}")
    print(f"Overall Score:    {result.overall_readiness_pct}%")
    print(f"Tickets Analyzed: {result.tickets_analyzed}")
    print(f"Open Bugs:        {result.open_bugs}")
    print(f"Resolved:         {result.resolved_bugs}")
    print()
    print("Dimension Breakdown:")
    print("-" * 50)
    for dim, detail in result.dimension_details.items():
        bar_len = int(detail["score"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {dim:30s}  {bar}  {detail['score']:5.1f}%  ({detail['tickets_affected']} tickets)")
    print()
    print("Suggestions:")
    print("-" * 50)
    for i, s in enumerate(result.suggestions, 1):
        print(f"  {i}. {s}")
    print()
