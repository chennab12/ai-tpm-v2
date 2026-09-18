# AI-TPM-v2

AI Inference Benchmarking & GPU Readiness Analysis Platform.

Test and benchmark standard AIML models on GPU (especially Intel), track TPM metrics
(TTFT, TPOT, TPS, P50/P95 latency), analyze Jira tickets for GPU model readiness,
and generate executive-friendly decisions.

## Roadmap

```
V1  Basic inference
V2  Benchmarking            <-- implemented
V3  RAG                     <-- implemented
V4  Jira ticket analyzer    <-- implemented
V5  Tool-calling agent      <-- implemented
V6  Bug triage agent        <-- implemented
V7  Reproduction agent      <-- implemented
V8  Fix recommendation      <-- implemented
V9  Automatic validation    <-- implemented
V10 TPM dashboard           <-- implemented
```

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install
uv sync

# Or with pip
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate synthetic Jira tickets
python -m src.cli synthetic

# 2. Run a benchmark
python -m src.cli benchmark --model distilgpt2 --runs 5

# 3. Analyze GPU readiness
python -m src.cli readiness --model Llama-3-8B

# 4. Triage tickets
python -m src.cli triage --key BUG-101

# 5. Start API server
python -m src.cli api

# 6. Start dashboard
python -m src.cli dashboard

# 7. Query the knowledge base (RAG)
python -m src.cli rag "why is ttft important"

# 8. Ask the tool-calling agent
python -m src.cli agent "what is the readiness of llama-3-8b"
```

## Docker

```bash
docker compose up --build   # CPU-only baseline (works on any host)
```

- API: http://localhost:8000
- MLflow: http://localhost:5000
- Dashboard: http://localhost:8501

GPU passthrough is opt-in via an override file (the base compose runs CPU
only, so it works on machines without the NVIDIA container toolkit):

```bash
# NVIDIA hosts (requires nvidia-container-toolkit) - uncomment the NVIDIA block
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up

# Intel hosts - uncomment the Intel block and ensure /dev/dri exists
# (image needs the Intel oneAPI runtime for SYCL/Level-Zero)
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token - how long until response starts |
| TPOT | Time per output token |
| TPS | Tokens per second (throughput) |
| P50 latency | Typical user experience |
| P95 latency | Slow-user experience |
| CPU/RAM | Host-side resource monitoring |
| GPU util | GPU utilization & memory |

## Acceptance Scorecard

Every benchmark produces a PASS/WARN/FAIL scorecard:

```json
{
  "ttft":        {"status": "PASS", "value": 0.5, "target": "<=1.0s"},
  "p95_latency": {"status": "PASS", "value": 8.0, "target": "<=10s"},
  "tps":         {"status": "PASS", "value": 25.0, "target": ">=5"},
  "tpot":        {"status": "PASS", "value": 40.0, "target": "<=200ms"},
  "error_rate":  {"status": "PASS", "value": 0.0, "target": "<=1%"},
  "cpu":         {"status": "PASS", "value": 45.0, "target": "<=90%"},
  "ram":         {"status": "PASS", "value": 60.0, "target": "<=85%"},
  "overall": "PASS"
}
```

## GPU Readiness

Analyzes Jira tickets (synthetic or real) to answer:

- Is the GPU Model Ready? And by how much percent?
- Which dimensions are below target?
- What should be improved to reach Production Ready?

Returns readiness score 0-100 with level:

| Score | Level |
|-------|-------|
| 80-100 | Production Ready |
| 60-79 | Near Ready |
| 40-59 | Needs Work |
| 0-39 | Not Ready |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/benchmark` | Run benchmark (JSON body) |
| GET | `/readiness` | GPU readiness analysis |
| POST | `/triage` | Triage a ticket |
| GET | `/tickets` | List synthetic tickets |
| GET | `/tickets/{key}` | Get specific ticket |
| GET | `/models/search` | Search HuggingFace |
| GET | `/models/{id}` | HF model info |
| GET | `/logs/scorecard` | Latest scorecard |
| GET | `/logs/results` | CSV results |
| POST | `/generate/synthetic-data` | Generate tickets |
| POST | `/rag/query` | RAG question against knowledge base |
| POST | `/agent/ask` | Tool-calling agent (natural language) |

## RAG Knowledge Base (V3)

Retrieval-augmented answers grounded in local docs
(`data/knowledge_base/*.md` - benchmark metrics, GPU readiness, Intel GPU
guide, vLLM tuning). Uses a lightweight numpy TF-IDF + cosine index, with
optional generative answers when a local model is supplied:

```bash
python -m src.cli rag "how do I fix high TTFT"
python -m src.cli rag "why did my uptime drop after the runtime upgrade" --model distilgpt2
```

Tool-calling agent (V5) routes requests like "run a benchmark for X",
"what is the readiness of Y", "show tickets", "search models for Z" to the
right platform tool automatically.

## Components

- **Benchmark engine** - streaming inference, TTFT/TPOT/TPS, P50/P95 aggregation
- **MLflow tracking** - experiment logging & comparison
- **Jira analyzer** - GPU readiness scoring with 6 weighted dimensions
- **Agentic AI** - bug triage, fix recommendation, reproduction plans, tool-calling agent
- **RAG** - TF-IDF knowledge base retrieval + optional generative answers
- **HuggingFace connector** - model discovery & metadata
- **Observability** - Prometheus metrics + structured logging
- **FastAPI server** - REST API for all components
- **Dashboard** - Streamlit TPM dashboard

## Example Ticket Analysis (BUG-101)

Given a customer ticket:
- TTFT: 2.4s observed vs 1.0s expected
- GPU util: 47%, Memory: 82%
- Batch: 8, Concurrency: 16

The system:
1. Triages severity (P1, Performance Regression)
2. Hypothesizes root cause (prefill overhead / KV cache fragmentation)
3. Generates fix options (prefix caching, batch tuning, profile)
4. Creates reproduction plan
5. Computes readiness impact across dimensions
6. Suggests improvement priorities