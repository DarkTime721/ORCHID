# Single / Multi Agent Pipeline

A LangGraph-based agent routing pipeline that automatically decides whether a task
is best handled by a **single agent** or a **multi-agent** architecture, then
evaluates the result through a judge and exposes everything via **FastAPI**.

---

## Project Structure

```
project/
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
