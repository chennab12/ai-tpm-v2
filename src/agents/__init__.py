"""Agentic AI components for bug triage, fix recommendation, and validation."""

import json
import re
from dataclasses import dataclass

from .tool_agent import AgentResponse, Tool, ToolCall, ToolCallingAgent

__all__ = [
    "BugTriageAgent",
    "TriageResult",
    "FixRecommendationAgent",
    "FixRecommendation",
    "ReproductionAgent",
    "ReproductionPlan",
    "ToolCallingAgent",
    "Tool",
    "ToolCall",
    "AgentResponse",
]


# ============================================================
# Bug Triage Agent
# ============================================================

@dataclass
class TriageResult:
    ticket_key: str
    severity: str  # P0, P1, P2, P3
    category: str
    root_cause_hypothesis: str
    affected_components: list[str]
    recommended_assignee_role: str
    estimated_effort: str
    blocking: bool


class BugTriageAgent:
    """Automatically triage inference-related bug tickets."""

    SEVERITY_RULES = [
        (["oom", "crash", "cuda error", "empty-response", "kernel launch"], "P0", "Service Down"),
        (["regression", "degraded", "worse"], "P1", "Performance Regression"),
        (["ttft", "latency", "slow"], "P2", "Performance Degradation"),
        (["configuration", "setup", "install"], "P3", "Configuration Issue"),
    ]

    COMPONENT_MAP = {
        "oom": "GPU Memory Manager",
        "kv-cache": "KV Cache Allocator",
        "ttft": "Prefill Engine",
        "tps": "Decode Scheduler",
        "batch": "Batch Controller",
        "concurrency": "Request Router",
        "gpu-detection": "Device Manager",
        "quantization": "Model Loader",
        "intel": "Hardware Abstraction Layer",
        "tensorrt": "TRT Compiler",
    }

    def triage(self, ticket: dict) -> TriageResult:
        """Triage a single ticket."""
        text = f"{ticket.get('summary', '')} {ticket.get('description', '')}".lower()

        severity = "P3"
        category = "Unknown"
        for keywords, sev, cat in self.SEVERITY_RULES:
            if any(kw in text for kw in keywords):
                severity, category = sev, cat
                break

        components = []
        for kw, comp in self.COMPONENT_MAP.items():
            if kw in text:
                components.append(comp)

        # Root cause hypothesis
        hypothesis = self._hypothesize(text, ticket)

        # Effort estimation
        effort = "S" if severity in ("P2", "P3") else "M" if severity == "P1" else "L"

        # Assignee role
        role = "GPU Engineer" if any(k in text for k in ["cuda", "gpu", "intel", "sycl"]) else "ML Engineer"

        return TriageResult(
            ticket_key=ticket.get("key", "UNKNOWN"),
            severity=severity,
            category=category,
            root_cause_hypothesis=hypothesis,
            affected_components=components or ["General"],
            recommended_assignee_role=role,
            estimated_effort=effort,
            blocking=(severity in ("P0", "P1")),
        )

    def _hypothesize(self, text: str, ticket: dict) -> str:
        """Generate a root cause hypothesis."""
        if "oom" in text or "failed to allocate" in text:
            return "Memory allocation exceeds available GPU VRAM. Likely caused by batch size or sequence length exceeding memory budget."
        if ("kv-cache" in text or "kv cache" in text) and ("exhaust" in text or "fill" in text):
            return "KV cache grows unboundedly causing memory exhaustion. Requires cache eviction policy or memory budget limits."
        if "quantiz" in text:
            return "Quantization may be introducing excessive overhead in dequantization or losing critical precision."
        if "ttft" in text or "cold-start" in text:
            return "TTFT regression likely due to prefill computation overhead. Check for KV cache fragmentation, engine recompilation, or attention implementation changes."
        if "concurrency" in text and "empty" in text:
            return "Request handling under concurrency may have race conditions or resource contention in the scheduler."
        if "intel" in text or "arc" in text or "sycl" in text:
            return "Intel GPU backend may lack full operator coverage. SYCL kernel compatibility needs verification."
        if "regression" in text or ("increased" in text and "ms" in text):
            return "Performance regression suggests a code change affected the hot path. Bisect recent runtime commits."
        if "128 tokens" in text or "stops at" in text:
            return "Generation stops prematurely. Likely a stop-condition or max_tokens configuration bug rather than hardware issue."
        return "Requires manual investigation of logs and profiling data."


# ============================================================
# Fix Recommendation Agent
# ============================================================

@dataclass
class FixRecommendation:
    ticket_key: str
    root_cause: str
    fix_options: list[dict]
    confidence: float
    test_plan: list[str]


class FixRecommendationAgent:
    """Suggest fix options based on ticket analysis."""

    FIX_DATABASE = {
        "oom": [
            {"action": "Reduce batch size", "config": "max_num_batched_tokens", "impact": "High", "effort": "Low"},
            {"action": "Enable quantization", "config": "quantization=AWQ", "impact": "High", "effort": "Medium"},
            {"action": "Enable tensor parallelism", "config": "tensor_parallel_size=2", "impact": "High", "effort": "Medium"},
            {"action": "Set max_model_len limit", "config": "max_model_len=4096", "impact": "Medium", "effort": "Low"},
        ],
        "ttft": [
            {"action": "Enable prefix caching", "config": "enable_prefix_caching=True", "impact": "High", "effort": "Low"},
            {"action": "Reduce context window", "config": "max_model_len", "impact": "Medium", "effort": "Low"},
            {"action": "Profile with nsight", "config": "NVIDIA_Nsight_Compute", "impact": "High", "effort": "Medium"},
            {"action": "Check prefill chunking", "config": "max_num_batched_tokens", "impact": "Medium", "effort": "Low"},
        ],
        "tps": [
            {"action": "Increase batch size", "config": "max_num_seqs=64", "impact": "High", "effort": "Low"},
            {"action": "Enable speculative decoding", "config": "speculative_model", "impact": "Medium", "effort": "High"},
            {"action": "Use Flash Attention", "config": "enforce_eager=False", "impact": "Medium", "effort": "Low"},
            {"action": "Enable CUDA graphs", "config": "enforce_eager=False", "impact": "Medium", "effort": "Low"},
        ],
        "concurrency": [
            {"action": "Enable request scheduling", "config": "scheduling_policy=fcfs", "impact": "High", "effort": "Low"},
            {"action": "Increase queue size", "config": "max_queue_size=2048", "impact": "Medium", "effort": "Low"},
            {"action": "Add rate limiting", "config": "rate_limit", "impact": "Medium", "effort": "Medium"},
        ],
        "intel-gpu": [
            {"action": "Install Intel oneAPI toolkit", "config": "oneAPI_basekit", "impact": "High", "effort": "Medium"},
            {"action": "Use IPEX backend", "config": "intel-extension-for-pytorch", "impact": "High", "effort": "Medium"},
            {"action": "Verify SYCL support", "config": "intel-sycl", "impact": "High", "effort": "Low"},
            {"action": "Use OpenVINO backend", "config": "openvino", "impact": "High", "effort": "High"},
        ],
        "kv-cache": [
            {"action": "Set KV cache budget", "config": "gpu_memory_utilization=0.9", "impact": "High", "effort": "Low"},
            {"action": "Enable PagedAttention", "config": "default in vLLM", "impact": "High", "effort": "Low"},
            {"action": "Add eviction policy", "config": "kv_cache_block_size", "impact": "Medium", "effort": "Medium"},
        ],
    }

    def recommend(self, triage: TriageResult, ticket: dict) -> FixRecommendation:
        """Generate fix recommendations based on triage."""
        text = f"{ticket.get('summary', '')} {ticket.get('description', '')}".lower()

        root_cause = triage.root_cause_hypothesis
        options = []

        # Match fix database entries
        for key, fixes in self.FIX_DATABASE.items():
            if key in text or key in " ".join(triage.affected_components).lower():
                options.extend(fixes)

        if not options:
            options = [{"action": "Manual investigation required", "config": "N/A", "impact": "Unknown", "effort": "Unknown"}]

        # Test plan
        test_plan = [
            f"1. Run warm-up inference (3 iterations) on {ticket.get('model', 'target')} model",
            f"2. Benchmark with input_tokens={ticket.get('input_tokens', 1024)}, output_tokens={ticket.get('output_tokens', 256)}",
            "3. Measure TTFT, TPOT, TPS against baseline",
            "4. Monitor GPU/CPU/RAM utilization during test",
            "5. Run stability test for 1+ hour under production concurrency",
        ]

        confidence = 0.8 if len(options) > 1 else 0.5

        return FixRecommendation(
            ticket_key=triage.ticket_key,
            root_cause=root_cause,
            fix_options=options[:5],
            confidence=confidence,
            test_plan=test_plan,
        )


# ============================================================
# Reproduction Agent
# ============================================================

@dataclass
class ReproductionPlan:
    ticket_key: str
    environment: dict
    steps: list[str]
    expected_vs_actual: dict
    synthetic_data: dict


class ReproductionAgent:
    """Generate reproduction plans from ticket data."""

    def create_plan(self, ticket: dict) -> ReproductionPlan:
        """Create a structured reproduction plan."""
        return ReproductionPlan(
            ticket_key=ticket.get("key", "UNKNOWN"),
            environment={
                "model": ticket.get("model", "unknown"),
                "runtime": ticket.get("runtime", "unknown"),
                "gpu": ticket.get("gpu_type", "NVIDIA A100"),
                "cuda_version": "12.1",
            },
            steps=[
                f"Deploy {ticket.get('model', 'model')} with {ticket.get('runtime', 'runtime')}",
                f"Set input_tokens={ticket.get('input_tokens', 1024)}, output_tokens={ticket.get('output_tokens', 256)}",
                f"Set batch_size={ticket.get('batch_size', 1)}, concurrency={ticket.get('concurrency', 1)}",
                "Run warm-up inference (3 iterations)",
                f"Execute benchmark: {ticket.get('input_tokens', 1024)} input tokens, {ticket.get('output_tokens', 256)} output tokens",
                "Monitor GPU utilization, memory, TTFT, TPOT, TPS",
                "Compare against expected values from ticket",
            ],
            expected_vs_actual={
                "ttft_expected_ms": ticket.get("ttft_expected_ms", "N/A"),
                "ttft_observed_ms": ticket.get("ttft_observed_ms", "N/A"),
                "tps_expected": ticket.get("tps_expected", "N/A"),
                "tps_observed": ticket.get("tps_observed", "N/A"),
            },
            synthetic_data={
                "prompt": "Explain the concept of technical program management in AI/ML projects.",
                "input_token_count": ticket.get("input_tokens", 1024),
                "max_new_tokens": ticket.get("output_tokens", 256),
                "batch_size": ticket.get("batch_size", 1),
                "concurrency": ticket.get("concurrency", 1),
            },
        )
