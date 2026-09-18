"""FastAPI server for AI-TPM benchmarking platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from src.config import DATA_DIR, LOGS_DIR, BenchmarkConfig, settings

# ============================================================
# App
# ============================================================

app = FastAPI(
    title="AI-TPM Benchmark API",
    description="AI Inference Benchmarking & GPU Readiness Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response models
# ============================================================


class BenchmarkRequest(BaseModel):
    model_name: str = "distilgpt2"
    max_new_tokens: int = 128
    num_runs: int = 5
    warmup_runs: int = 1
    prompt: str = "Explain why AI inference latency matters to a TPM."


class ReadinessRequest(BaseModel):
    model: str = ""
    ticket_file: str = ""


class RagQueryRequest(BaseModel):
    question: str
    model: str = ""
    max_new_tokens: int = 120


class AgentAskRequest(BaseModel):
    query: str
    default_model: str = ""


# ============================================================
# Routes
# ============================================================


@app.get("/")
def root():
    return {"service": "ai-tpm-api", "version": "0.1.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    from src.observability import get_metrics_text
    return get_metrics_text()


@app.post("/benchmark")
def run_benchmark(req: BenchmarkRequest):
    """Run a full benchmark session."""
    from src.benchmark.engine import run_benchmark, print_summary
    from src.observability import BENCHMARK_RUNS, logger

    config = BenchmarkConfig(
        model_name=req.model_name,
        max_new_tokens=req.max_new_tokens,
        num_runs=req.num_runs,
        warmup_runs=req.warmup_runs,
        prompt=req.prompt,
    )

    logger.info(f"Starting benchmark: model={req.model_name} runs={req.num_runs}")
    summary = run_benchmark(config)
    BENCHMARK_RUNS.labels(model=req.model_name).inc()
    print_summary(summary)

    return {
        "model": summary.model_name,
        "device": summary.device,
        "runs": f"{summary.successful_runs}/{summary.num_runs}",
        "avg_latency_sec": summary.avg_latency_sec,
        "p50_latency_sec": summary.p50_latency_sec,
        "p95_latency_sec": summary.p95_latency_sec,
        "avg_ttft_sec": summary.avg_ttft_sec,
        "avg_tpot_sec": summary.avg_tpot_sec,
        "avg_tps": summary.avg_tps,
        "error_rate_percent": summary.error_rate_percent,
        "avg_cpu_percent": summary.avg_cpu_percent,
        "avg_ram_percent": summary.avg_ram_percent,
        "avg_gpu_utilization": summary.avg_gpu_utilization,
        "scorecard": summary.scorecard,
    }


@app.get("/readiness")
def get_readiness(model: str = Query(default="")):
    """Get GPU readiness score for a model."""
    from src.jira_analyzer import load_and_analyze, print_readiness
    from src.observability import READINESS_SCORE, logger

    result = load_and_analyze(model=model)
    print_readiness(result)

    # Update Prometheus gauges
    for dim, score in result.dimension_scores.items():
        READINESS_SCORE.labels(model=result.model, dimension=dim).set(score)

    return {
        "model": result.model,
        "overall_readiness_pct": result.overall_readiness_pct,
        "readiness_level": result.readiness_level,
        "dimension_scores": result.dimension_scores,
        "tickets_analyzed": result.tickets_analyzed,
        "open_bugs": result.open_bugs,
        "resolved_bugs": result.resolved_bugs,
        "suggestions": result.suggestions,
    }


@app.post("/triage")
def triage_ticket(ticket: dict[str, Any]):
    """Triage a single ticket."""
    from src.agents import BugTriageAgent, FixRecommendationAgent, ReproductionAgent

    triage_agent = BugTriageAgent()
    fix_agent = FixRecommendationAgent()
    repro_agent = ReproductionAgent()

    triage = triage_agent.triage(ticket)
    fix = fix_agent.recommend(triage, ticket)
    repro = repro_agent.create_plan(ticket)

    return {
        "triage": {
            "ticket_key": triage.ticket_key,
            "severity": triage.severity,
            "category": triage.category,
            "root_cause_hypothesis": triage.root_cause_hypothesis,
            "affected_components": triage.affected_components,
            "recommended_assignee_role": triage.recommended_assignee_role,
            "estimated_effort": triage.estimated_effort,
            "blocking": triage.blocking,
        },
        "fix_recommendation": {
            "root_cause": fix.root_cause,
            "fix_options": fix.fix_options,
            "confidence": fix.confidence,
            "test_plan": fix.test_plan,
        },
        "reproduction_plan": {
            "environment": repro.environment,
            "steps": repro.steps,
            "expected_vs_actual": repro.expected_vs_actual,
            "synthetic_data": repro.synthetic_data,
        },
    }


@app.get("/tickets")
def list_tickets():
    """List synthetic tickets."""
    ticket_file = DATA_DIR / "synthetic_tickets.json"
    if not ticket_file.exists():
        from src.synthetic import save_synthetic_data
        save_synthetic_data()
    with open(ticket_file) as f:
        return json.load(f)


@app.get("/tickets/{ticket_key}")
def get_ticket(ticket_key: str):
    """Get a specific ticket."""
    ticket_file = DATA_DIR / "synthetic_tickets.json"
    if not ticket_file.exists():
        raise HTTPException(status_code=404, detail="No tickets found. Generate synthetic data first.")
    with open(ticket_file) as f:
        tickets = json.load(f)
    for t in tickets:
        if t.get("key") == ticket_key:
            return t
    raise HTTPException(status_code=404, detail=f"Ticket {ticket_key} not found")


@app.get("/logs/scorecard")
def get_latest_scorecard():
    """Get the latest scorecard."""
    sc_file = LOGS_DIR / "scorecard.json"
    if not sc_file.exists():
        return {"message": "No scorecard yet. Run a benchmark first."}
    with open(sc_file) as f:
        return json.load(f)


@app.get("/logs/results")
def get_csv_results():
    """Get benchmark results from CSV."""
    csv_file = LOGS_DIR / "inference_results.csv"
    if not csv_file.exists():
        return {"message": "No results yet. Run a benchmark first."}
    import csv
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        return list(reader)


@app.get("/models/search")
def search_models(query: str = Query(default="llama"), limit: int = 10):
    """Search HuggingFace Hub for models."""
    from src.huggingface_connector import HuggingFaceConnector
    connector = HuggingFaceConnector(token=settings.HF_TOKEN)
    return connector.search_models(query, limit=limit)


@app.get("/models/{model_id:path}")
def get_model_info(model_id: str):
    """Get HuggingFace model info."""
    from src.huggingface_connector import HuggingFaceConnector
    connector = HuggingFaceConnector(token=settings.HF_TOKEN)
    info = connector.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found on HuggingFace")
    estimate = connector.get_model_size_estimate(model_id)
    return {"info": info.__dict__, "estimate": estimate}


@app.post("/verdict")
def model_verdict(req: BenchmarkRequest, model: str = Query("", description="Model filter for readiness")):
    """
    Combined Model Readiness Verdict.

    Runs a real benchmark, computes readiness from tickets, and produces
    an executive summary: is the GPU model ready, by what percent, and
    what to improve.
    """
    from src.benchmark.engine import run_benchmark
    from src.jira_analyzer import load_and_analyze
    from src.tracking import log_benchmark
    from src.observability import logger

    logger.info(f"Model verdict requested: model={req.model_name}")

    # 1. Run benchmark
    config = BenchmarkConfig(
        model_name=req.model_name,
        max_new_tokens=req.max_new_tokens,
        num_runs=req.num_runs,
        warmup_runs=req.warmup_runs,
        prompt=req.prompt,
    )
    summary = run_benchmark(config)
    log_benchmark(summary)

    # 2. Readiness from tickets (filter by model if provided)
    filter_model = model or (tickets_get_model(req.model_name))
    readiness = load_and_analyze(model=filter_model or "")

    # 3. Build verdict - combine benchmark scorecard + readiness
    bench_status = summary.scorecard.get("overall", "UNKNOWN")
    readiness_score = readiness.overall_readiness_pct

    # Composite score: blend benchmark outcome and ticket-based readiness
    composite = readiness_score
    if bench_status == "PASS":
        composite = min(100, readiness_score + 10)
    elif bench_status == "FAIL":
        composite = max(0, readiness_score - 15)
    composite = round(composite, 1)

    if composite >= 80:
        verdict = "PRODUCTION READY"
    elif composite >= 60:
        verdict = "NEAR READY - CONDITIONAL APPROVAL"
    elif composite >= 40:
        verdict = "NEEDS WORK - NOT YET READY"
    else:
        verdict = "NOT READY - SIGNIFICANT GAPS"

    return {
        "verdict": verdict,
        "composite_readiness_pct": composite,
        "benchmark_result": {
            "runs_ok": f"{summary.successful_runs}/{summary.num_runs}",
            "device": summary.device,
            "avg_ttft_sec": summary.avg_ttft_sec,
            "p95_latency_sec": summary.p95_latency_sec,
            "avg_tps": summary.avg_tps,
            "avg_tpot_ms": round(summary.avg_tpot_sec * 1000, 1),
            "scorecard": summary.scorecard,
        },
        "ticket_readiness": {
            "level": readiness.readiness_level,
            "score_pct": readiness_score,
            "tickets_analyzed": readiness.tickets_analyzed,
            "open_bugs": readiness.open_bugs,
            "dimension_scores": readiness.dimension_scores,
        },
        "suggestions": readiness.suggestions,
        "executive_summary": (
            f"Model {req.model_name} on {summary.device} is {verdict} "
            f"at {composite}% readiness. "
            f"Benchmark scorecard: {bench_status}. "
            f"Ticket-based readiness: {readiness_score}% ({readiness.readiness_level}). "
        ),
    }


@app.post("/rag/query")
def rag_query_endpoint(req: RagQueryRequest):
    """Answer a question against the project knowledge base (RAG)."""
    from src.rag.pipeline import rag_query

    answer = rag_query(
        req.question,
        model=req.model or None,
        max_new_tokens=req.max_new_tokens,
    )
    return {
        "question": answer.question,
        "answer": answer.answer,
        "model": answer.model,
        "evidence": answer.retrieved_context,
    }


@app.post("/agent/ask")
def agent_ask(req: AgentAskRequest):
    """Route a natural-language request to the platform tool agent."""
    from src.agents import ToolCallingAgent

    agent = ToolCallingAgent()
    response = agent.ask(req.query)
    return {
        "query": response.query,
        "intent": response.intent,
        "confidence": response.confidence,
        "summary": response.summary,
        "calls": [
            {
                "tool": c.tool,
                "args": c.args,
                "result": c.result,
            }
            for c in response.calls
        ],
    }


def tickets_get_model(model_id: str) -> str:
    """Map a HF model id to a friendly ticket model name."""
    from src.huggingface_connector import HuggingFaceConnector
    info = HuggingFaceConnector().get_model_info(model_id)
    if not info:
        return ""
    return info.model_id.split("/")[-1]


@app.post("/generate/synthetic-data")
def generate_data():
    """Generate synthetic Jira tickets for testing."""
    from src.synthetic import save_synthetic_data
    tickets = save_synthetic_data()
    return {"generated": len(tickets), "path": str(DATA_DIR / "synthetic_tickets.json")}
