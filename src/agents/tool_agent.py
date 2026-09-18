"""
V5 - Tool-Calling Agent.

Routes natural-language requests to a registry of platform tools
(benchmark, readiness, tickets, triage, HF search, scorecard, RAG)
by matching intent patterns and extracting parameters. Runs offline
so it can be demonstrated without an external LLM, but each tool
wraps the real platform capability.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """A registered tool the agent can invoke."""

    name: str
    description: str
    intent_keywords: list[str]
    handler: Callable[..., Any]
    example: str = ""


@dataclass
class ToolCall:
    """Record of one agent tool invocation."""

    tool: str
    args: dict[str, Any]
    result: Any
    raw_query: str
    confidence: float = 0.0


@dataclass
class AgentResponse:
    """Agent output for a natural-language request."""

    query: str
    intent: str
    calls: list[ToolCall] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0


def _extract_model(query: str, default: str = "") -> str:
    """Extract a model identifier observed in the query (lowercased match)."""
    m = re.search(r"\b(mistral-7b|llama[-\s]?[0-9.\-]*[a-z]*|gemma[-\s]?[0-9]+|persian-?gpt|qwen[-\s]?[0-9]*|phi[-\s]?[0-9]*|distilgpt2|gpt-?2?)\b", query, re.I)
    if m:
        return m.group(1).lower()
    return default


def _extract_int(query: str, default: int = 0) -> int:
    """Extract an integer from the query, if any."""
    m = re.search(r"(\d+)", query)
    return int(m.group(1)) if m else default


class ToolCallingAgent:
    """Routes natural-language queries to registered platform tools."""

    def __init__(self, tools: list[Tool] | None = None):
        self.tools = tools or _default_tools()

    def route(self, query: str) -> str:
        """Return the best-matching tool name for a query."""
        q = query.lower()
        best = None
        best_hits = 0
        for tool in self.tools:
            hits = sum(1 for kw in tool.intent_keywords if kw in q)
            if hits > best_hits:
                best_hits, best = hits, tool
        return best.name if best and best_hits > 0 else "help"

    def ask(self, query: str) -> AgentResponse:
        """Execute the routed tool and return a response."""
        tool_name = self.route(query)
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if tool is None:
            return AgentResponse(query, "help", summary="I can run benchmarks, check readiness, list tickets, triage bugs, search HuggingFace models, or query the knowledge base. Try: 'run a benchmark for distilgpt2' or 'what is the readiness of Llama-3-8B?'")

        args = _route_args(tool_name, query)
        try:
            result = tool.handler(**args)
        except Exception as exc:
            result = {"error": str(exc)}

        return AgentResponse(
            query=query,
            intent=tool_name,
            calls=[ToolCall(tool=tool_name, args=args, result=result, raw_query=query, confidence=0.9)],
            summary=_summarize(tool_name, args, result),
        )


# ============================================================
# Argument extraction per tool
# ============================================================


def _route_args(tool_name: str, query: str) -> dict[str, Any]:
    if tool_name == "benchmark":
        return {"model_name": _extract_model(query, "distilgpt2")}
    if tool_name == "readiness":
        return {"model": _extract_model(query, "")}
    if tool_name == "ticket":
        m = re.search(r"\b(BUG-\d+)\b", query, re.I)
        return {"ticket_key": m.group(1).upper() if m else ""}
    if tool_name in ("triage", "search", "rag"):
        return {"query": query}
    return {}


# ============================================================
# Tool handlers
# ============================================================


def _handle_benchmark(model_name: str) -> dict:
    from src.benchmark.engine import run_benchmark
    from src.config import BenchmarkConfig

    summary = run_benchmark(BenchmarkConfig(model_name=model_name, max_new_tokens=64, num_runs=2, warmup_runs=1))
    return {
        "model": summary.model_name,
        "device": summary.device,
        "avg_ttft_sec": summary.avg_ttft_sec,
        "avg_tps": summary.avg_tps,
        "p95_latency_sec": summary.p95_latency_sec,
        "scorecard": summary.scorecard.get("overall"),
        "runs_ok": f"{summary.successful_runs}/{summary.num_runs}",
    }


def _handle_readiness(model: str) -> dict:
    from src.jira_analyzer import load_and_analyze

    result = load_and_analyze(model=model)
    return {
        "model": result.model,
        "score_pct": result.overall_readiness_pct,
        "level": result.readiness_level,
        "tickets_analyzed": result.tickets_analyzed,
        "suggestions": result.suggestions[:3],
    }


def _handle_ticket(ticket_key: str) -> dict:
    import json
    from src.config import DATA_DIR

    path = DATA_DIR / "synthetic_tickets.json"
    if not path.exists():
        return {"error": "No tickets. Run: python -m src.cli synthetic"}
    with open(path) as f:
        tickets = json.load(f)
    if ticket_key:
        for t in tickets:
            if t.get("key") == ticket_key:
                return {"ticket": t}
        return {"error": f"Ticket {ticket_key} not found"}
    return {"count": len(tickets), "keys": [t["key"] for t in tickets]}


def _handle_triage(query: str) -> dict:
    import json
    from pathlib import Path
    from src.agents import BugTriageAgent, FixRecommendationAgent

    key = _extract_int(query) or 101
    path = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic_tickets.json"
    with open(path) as f:
        tickets = json.load(f)
    ticket = next((t for t in tickets if _extract_int(t["key"]) == key), tickets[0])
    if _extract_int(ticket["key"]) != key:
        ticket["key"] = f"BUG-{key}"
    triage = BugTriageAgent().triage(ticket)
    fix = FixRecommendationAgent().recommend(triage, ticket)
    return {
        "ticket": ticket.get("key"),
        "severity": triage.severity,
        "category": triage.category,
        "root_cause": triage.root_cause_hypothesis,
        "top_fix": fix.fix_options[0] if fix.fix_options else None,
    }


def _handle_search(query: str) -> dict:
    from src.huggingface_connector import HuggingFaceConnector

    terms = re.sub(r"(?i)^(search|find|models|for|on|hub)", " ", query).strip()
    results = HuggingFaceConnector().search_models(terms or query, limit=5)
    return {"results": results[:3]}


def _handle_scorecard() -> dict:
    import json
    from src.config import LOGS_DIR

    path = LOGS_DIR / "scorecard.json"
    if not path.exists():
        return {"error": "No scorecard yet. Run a benchmark first."}
    with open(path) as f:
        return json.load(f)


def _handle_rag(query: str) -> dict:
    from src.rag.pipeline import rag_query

    answer = rag_query(query)
    return {
        "answer": answer.answer,
        "evidence": answer.retrieved_context[:2],
    }


# ============================================================
# Default tool list
# ============================================================


def _default_tools() -> list[Tool]:
    return [
        Tool("benchmark", "Run an inference benchmark", ["benchmark", "bench", "inference", "run a"], _handle_benchmark,
             "run a benchmark for mistral-7b"),
        Tool("readiness", "GPU model readiness score", ["readiness", "ready", "percent", "% ready"], _handle_readiness,
             "what is the readiness of llama-3-8b"),
        Tool("ticket", "List or inspect Jira tickets", ["ticket", "bugs", "jira", "issue"], _handle_ticket,
             "show tickets"),
        Tool("triage", "Triage a bug ticket", ["triage", "severity", "root cause", "fix"], _handle_triage,
             "triage bug 101"),
        Tool("search", "Search HuggingFace Hub", ["search", "find model", "huggingface"], _handle_search,
             "search models for llama"),
        Tool("scorecard", "Latest benchmark scorecard", ["scorecard", "poc status", "result"], _handle_scorecard,
             "show the scorecard"),
        Tool("rag", "Ask the knowledge base", ["what is", "how do", "explain", "define", "tune", "why"], _handle_rag,
             "what is ttft"),
    ]


# ============================================================
# Summaries
# ============================================================


def _summarize(tool_name: str, args: dict, result: Any) -> str:
    if tool_name == "benchmark":
        return f"Benchmark for {args.get('model_name')}: scorecard {result.get('scorecard')}, TTFT {result.get('avg_ttft_sec')}s, TPS {result.get('avg_tps')}."
    if tool_name == "readiness":
        return f"Readiness for {result.get('model')}: {result.get('score_pct')}% ({result.get('level')})."
    if tool_name == "ticket":
        return f"Found {result.get('count', 1)} ticket(s)."
    if tool_name == "triage":
        return f"{result.get('ticket')} -> {result.get('severity')} ({result.get('category')}). Root cause: {result.get('root_cause', 'n/a')}"
    if tool_name == "search":
        return f"Found {len(result.get('results', []))} models on HuggingFace."
    if tool_name == "scorecard":
        return f"Overall scorecard status: {result.get('scorecard', {}).get('overall', 'n/a')}."
    return "Answered from the knowledge base."