"""CLI entrypoint for AI-TPM-v2."""

import argparse
import sys


def cmd_benchmark(args):
    from src.benchmark.engine import print_summary, run_benchmark
    from src.config import BenchmarkConfig

    print_summary(run_benchmark(
        BenchmarkConfig(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            num_runs=args.runs,
            warmup_runs=args.warmup,
        )
    ))
    return 0


def cmd_readiness(args):
    from src.jira_analyzer import load_and_analyze, print_readiness
    print_readiness(load_and_analyze(model=args.model))
    return 0


def cmd_synthetic(args):
    from src.synthetic import save_synthetic_data
    save_synthetic_data()
    return 0


def cmd_api(args):
    import uvicorn
    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=False)
    return 0


def cmd_dashboard(args):
    from src.dashboard.app import run_dashboard
    run_dashboard()
    return 0


def cmd_triage(args):
    import json
    from src.agents import BugTriageAgent, FixRecommendationAgent, ReproductionAgent
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent
    ticket_file = data_dir / "data" / "synthetic_tickets.json"
    with open(ticket_file) as f:
        tickets = json.load(f)

    if args.key:
        tickets = [t for t in tickets if t.get("key") == args.key]

    triage_agent = BugTriageAgent()
    fix_agent = FixRecommendationAgent()
    repro_agent = ReproductionAgent()

    for ticket in tickets:
        triage = triage_agent.triage(ticket)
        fix = fix_agent.recommend(triage, ticket)
        repro = repro_agent.create_plan(ticket)

        print(f"\n{'='*60}")
        print(f"  {ticket['key']}: {ticket['summary'][:60]}")
        print(f"{'='*60}")
        print(f"  Severity:            {triage.severity}")
        print(f"  Category:            {triage.category}")
        print(f"  Root cause:          {triage.root_cause_hypothesis}")
        print(f"  Components:          {', '.join(triage.affected_components)}")
        print(f"  Assignee role:       {triage.recommended_assignee_role}")
        print(f"  Estimated effort:    {triage.estimated_effort}")
        print(f"  Blocking:            {triage.blocking}")
        print(f"  Confidence:          {fix.confidence:.0%}")
        print(f"\n  Fix options:")
        for opt in fix.fix_options:
            print(f"    - {opt['action']:60s}  (impact: {opt['impact']}, effort: {opt['effort']})")
    return 0


def cmd_verdict(args):
    """Combined model readiness verdict."""
    import json
    from src.benchmark.engine import run_benchmark, print_summary
    from src.config import BenchmarkConfig
    from src.jira_analyzer import load_and_analyze, print_readiness

    summary = run_benchmark(BenchmarkConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.runs,
        warmup_runs=1,
    ))
    print_summary(summary)

    readiness = load_and_analyze(model=args.model)
    print_readiness(readiness)

    bench_status = summary.scorecard.get("overall", "UNKNOWN")
    composite = readiness.overall_readiness_pct
    if bench_status == "PASS":
        composite = min(100, composite + 10)
    elif bench_status == "FAIL":
        composite = max(0, composite - 15)
    composite = round(composite, 1)

    if composite >= 80:
        verdict = "PRODUCTION READY"
    elif composite >= 60:
        verdict = "NEAR READY - CONDITIONAL APPROVAL"
    elif composite >= 40:
        verdict = "NEEDS WORK - NOT YET READY"
    else:
        verdict = "NOT READY - SIGNIFICANT GAPS"

    print(f"\n{'='*65}")
    print(f"MODEL READINESS VERDICT  |  {args.model}")
    print(f"{'='*65}")
    print(f"  Benchmark scorecard:  {bench_status}")
    print(f"  Ticket readiness:     {readiness.overall_readiness_pct}% ({readiness.readiness_level})")
    print(f"  Composite readiness:  {composite}%")
    print(f"  VERDICT:              {verdict}")
    print(f"{'='*65}")
    print(f"\n  Top suggestions:")
    for i, s in enumerate(readiness.suggestions[:3], 1):
        print(f"   {i}. {s}")
    print()
    return 0


def cmd_rag(args):
    from src.rag.pipeline import rag_cli

    rag_cli(args.question, model=args.model or None)
    return 0


def cmd_agent(args):
    from src.agents import ToolCallingAgent

    response = ToolCallingAgent().ask(args.query)
    print(f"\nQuery: {response.query}")
    print(f"Intent: {response.intent}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"\nSummary: {response.summary}\n")
    for call in response.calls:
        print(f"Tool invoked: {call.tool} args={call.args}")
        print(f"  result: {call.result}\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ai-tpm",
        description="AI-TPM Benchmarking & GPU Readiness Platform",
    )
    sub = parser.add_subparsers(dest="command")

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Run model benchmark")
    p_bm.add_argument("--model", default="distilgpt2")
    p_bm.add_argument("--max-new-tokens", type=int, default=128)
    p_bm.add_argument("--runs", type=int, default=5)
    p_bm.add_argument("--warmup", type=int, default=1)
    p_bm.set_defaults(func=cmd_benchmark)

    # readiness
    p_rd = sub.add_parser("readiness", help="Analyze GPU readiness")
    p_rd.add_argument("--model", default="")
    p_rd.set_defaults(func=cmd_readiness)

    # synthetic
    p_syn = sub.add_parser("synthetic", help="Generate synthetic data")
    p_syn.set_defaults(func=cmd_synthetic)

    # api
    p_api = sub.add_parser("api", help="Start FastAPI server")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.set_defaults(func=cmd_api)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Start TPM dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    # triage
    p_t = sub.add_parser("triage", help="Analyze and triage tickets")
    p_t.add_argument("--key", default="")
    p_t.set_defaults(func=cmd_triage)

    # verdict
    p_v = sub.add_parser("verdict", help="Combined model readiness verdict (benchmark + tickets)")
    p_v.add_argument("--model", default="distilgpt2")
    p_v.add_argument("--max-new-tokens", type=int, default=128)
    p_v.add_argument("--runs", type=int, default=5)
    p_v.set_defaults(func=cmd_verdict)

    # rag
    p_rag = sub.add_parser("rag", help="Query the knowledge base with RAG")
    p_rag.add_argument("question", help="Question to answer from the knowledge base")
    p_rag.add_argument("--model", default="", help="Local model for generative answers (optional)")
    p_rag.set_defaults(func=cmd_rag)

    # agent
    p_ag = sub.add_parser("agent", help="Ask the tool-calling agent")
    p_ag.add_argument("query", help='Natural-language request, e.g. "what is the readiness of llama-3-8b"')
    p_ag.set_defaults(func=cmd_agent)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())