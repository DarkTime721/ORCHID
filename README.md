# ⬡ ORCHID — Orchestration Research for Comparing Heterogeneous Intelligent Designs
 
A full-stack AI engineering project that builds, evaluates, and visualizes an intelligent task routing system — empirically comparing single-agent vs multi-agent LLM architectures in real time.
 
---

## What it does
 
Most agentic AI systems hardcode whether to use one agent or many. ORCHID routes tasks dynamically:
 
- A **classifier node** analyzes the task and assigns a confidence score
- Tasks are routed to either a **single-agent pipeline** (direct LLM) or a **multi-agent pipeline** (parallel workers → synthesizer)
- A **judge node** scores both architectures on response quality (0–10)
- **10% of runs** trigger shadow evaluation — the non-routed architecture runs on the same task to produce ground-truth comparison data, eliminating survivor bias from the metrics dataset
- Every run is logged. A live dashboard shows latency, judge scores, routing distribution, and run history

---

## Architecture
 
```
Task Input
    │
    ▼
┌─────────────┐
│  Classifier │  ← task taxonomy + confidence scoring
└──────┬──────┘
       │
  ┌────┴────┐
  │         │
  ▼         ▼
Single    Parallel Workers
Agent     └──► Synthesizer
  │              │
  └──────┬───────┘
         ▼
    ┌─────────┐
    │  Judge  │  ← Llama 3.1 8B, structured output
    └────┬────┘
         ▼
  Optimal Decision

```
 
Shadow evaluation runs the losing branch on 10% of tasks independently, producing unbiased comparison data across both architectures.
 
---

## Stack
 
| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| Local inference | Ollama |
| Worker models | Qwen3 8B |
| Judge model | Llama 3.1 8B |
| Observability | LangSmith |
| Backend | FastAPI + SSE streaming |
| Frontend | React + Vite |
 
---
 
## Features
 
- **Live pipeline animation** — the UI shows which node is currently executing in real time via Server-Sent Events
- **Confidence-aware routing** — tasks below the confidence threshold are flagged as ambiguous
- **LLM-as-judge evaluation** — pairwise scoring with winner, per-architecture scores, and rationale
- **Shadow evaluation** — prevents survivor bias by occasionally running the non-routed architecture
- **Metrics dashboard** — avg latency, judge scores, and routing distribution across all runs
- **Run history** — full log of every task, routing decision, and judge result
---
 
## Hardware Note
 
This system runs entirely on local hardware — an RTX 4060 laptop GPU (8GB VRAM, 16GB RAM). Multiple LLM calls per run (classifier + agent(s) + judge) means inference is slow on constrained hardware, but the architecture is model-agnostic. This surfaces the real compute cost tradeoff between architectures that cloud-hosted benchmarks typically obscure.
Any Ollama-compatible model can be swapped in — larger or cloud-hosted models will significantly improve both speed and output quality.
 
**Recommended models given VRAM constraints:**
 
| Node | Model | Notes |
|---|---|---|
| Classifier | `qwen2.5` | Fast, reliable taxonomy |
| Worker | `qwen3:8b` | Primary worker model |
| Judge | `llama3.1` | No thinking mode, clean structured output |
 
---

## Project Structure

```
ORCHID/
│
├── main.py                    # FastAPI app — entry point
│
├── models/
│   ├── __init__.py
│   └── state.py               # TypedDicts: TaskState, ComplexityFeatures, RoutingHistorySummary
│
├── core/
│   ├── __init__.py
│   ├── utils.py               # count_message_tokens, append_to_json, tiktoken encoder
│   ├── history.py             # load_routing_history — reads run_history.json
│   ├── confidence.py          # calibrate, compute_single_score, compute_multi_scores,
│   │                          #   aggregate_multi_score, compute_confidence
│   ├── router.py              # confidence_router — routes to single / multi / both
│   └── decision.py            # optimal_decision — picks winner based on quality + cost
│
├── agents/
│   ├── __init__.py
│   ├── task_analysis.py       # TaskAnalysis Pydantic model, TASK_ANALYSIS_PROMPT,
│   │                          #   task_analysis_node (LLM + confidence computation)
│   ├── single_agent.py        # SingleAgentState, single_agent_subgraph (compiled)
│   ├── multi_agent.py         # MultiAgentState, WorkerState, worker_subgraph,
│   │                          #   multi_agent_subgraph (compiled)
│   └── judge.py               # Judge Pydantic model, JUDGE_SYSTEM_PROMPT, judge_node
│
├── pipeline/
│   ├── __init__.py
│   ├── graph.py               # input_node, output_node, main_graph (compiled)
│   ├── runner.py              # stream_pipeline — CLI-friendly streaming runner
│   └── shadow.py              # shadow_evaluation, shadow_run (10% shadow traffic)
│
└── requirements.txt
```

---

## Key Engineering Decisions
 
**Shadow evaluation for unbiased metrics** — only logging the routed architecture creates survivor bias. Running the non-routed path on 10% of tasks produces ground-truth comparison data regardless of routing decisions.
 
**Confidence-aware routing** — rather than a hard classifier, the system produces a confidence score and handles low-confidence cases as a separate ambiguous category, preventing forced misclassification.
 
**Llama 3.1 8B as judge** — chosen specifically because it has no thinking mode, making `.with_structured_output()` reliable without prompt hacks or response parsing workarounds.
 
**SSE over WebSockets** — simpler protocol for unidirectional server-to-client streaming. No handshake overhead, works through standard HTTP proxies, auto-reconnects natively in the browser.
 
---

## Installation

```bash
pip install -r requirements.txt
```

Ollama must be running locally with the required models pulled:

```bash
ollama pull qwen2.5
ollama pull qwen3:8b
ollama pull llama3.1
```

---

## Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs are available at `http://localhost:8000/docs`.

---

## API Endpoints

### `GET /health`
Liveness check.

```json
{ "status": "ok" }
```

---

### `POST /run`
Run the full pipeline and get a structured JSON response.

**Request body:**
```json
{ "task": "What is the boiling point of water at 8000 feet altitude?" }
```

**Response:**
```json
{
  "run_id": "...",
  "task": "...",
  "architecture_used": "single",
  "recommended_architecture": "single",
  "confidence": 0.87,
  "optimal_output": "...",
  "optimal_solution_architecture": "single",
  "decision_basis": "single_run",
  "judge_score_single": 88,
  "judge_score_multi": null,
  "judge_winner": "single",
  "score_differential": null,
  "judge_architecture_fit": "correct",
  "judge_explanation": "...",
  "single_tokens": 412,
  "multi_tokens": null,
  "total_tokens": 412,
  "llm_call_count": 3,
  "latency_ms": 4201.5,
  "success": true,
  "was_routing_correct": true
}
```

---

### `POST /stream`
Stream token-by-token output as `text/plain`.

**Request body:**
```json
{ "task": "Compare transformer, RWKV, and Mamba architectures" }
```

Returns a streaming plain-text response from the active agent node.

---

## Running via CLI (without FastAPI)

```python
from pipeline.runner import stream_pipeline

stream_pipeline("What is the time complexity of Dijkstra's algorithm and why?")
```

---

## Data Files

| File | Purpose |
|---|---|
| `run_history.json` | Append-only JSONL log of every completed run (used for routing feedback) |
| `shadow_run.json` | Append-only JSONL log of 10%-sampled shadow evaluations |

---

## Sample Tasks
 
```python
# Single-agent — self-contained, factual
"What is the time complexity of Dijkstra's algorithm and why?"
"Calculate the Schwarzschild radius of an object with the mass of Earth"
 
# Multi-agent — parallel research across multiple entities
"Compare the transformer architecture, RWKV, and Mamba — explain how each handles long-range dependencies"
"Find the latest benchmark scores for Llama 4, Qwen3, and Gemini 2.5 Pro across reasoning, coding, and math"
 
# Ambiguous — tests routing confidence
"Is RAG still relevant now that context windows are so large?"
"What should I know about deploying LLMs in production in 2025?"
```
 
---

## Module Responsibilities (quick reference)

| Module | Responsibility |
|---|---|
| `models/state.py` | All shared TypedDict state definitions |
| `core/history.py` | Reads past runs to compute routing adjustment |
| `core/confidence.py` | Pure functions — no I/O, fully testable |
| `core/router.py` | Stateless routing decision |
| `core/decision.py` | Stateless optimal-winner selection |
| `agents/task_analysis.py` | LLM call + confidence wiring |
| `agents/single_agent.py` | Single-agent LangGraph subgraph |
| `agents/multi_agent.py` | Orchestrator + worker + synthesizer subgraph |
| `agents/judge.py` | Evaluation LLM node |
| `pipeline/graph.py` | Assembles and compiles the main graph |
| `pipeline/runner.py` | CLI streaming helper |
| `pipeline/shadow.py` | Shadow traffic evaluation (10% sampling) |
| `main.py` | FastAPI app with `/health`, `/run`, `/stream` |
