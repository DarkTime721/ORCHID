import json
import uuid
import statistics
from collections import defaultdict
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.graph import main_graph
from pipeline.shadow import shadow_run
from core.utils import append_to_json

app = FastAPI(
    title="Single/Multi Agent Pipeline API",
    description="LangGraph-based agent routing pipeline exposed via FastAPI",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    task: str


def _load_history(filepath: str = "run_history.json") -> list[dict]:
    """Read all runs from the append-only JSONL log."""
    try:
        with open(filepath, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Events line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/run")
async def run_pipeline(request: TaskRequest):
    """
    Run the full pipeline and stream Server-Sent Events back to the frontend.

    Event shapes emitted:
      {"type": "metadata", "routing": "single"|"multi", "task_type": "...", "confidence": 0.87}
      {"type": "token",    "content": "...", "node": "single_agent_node"|"synthesizer"}
      {"type": "done",     "run_id": "..."}
      {"type": "error",    "message": "..."}
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    async def generate():
        task_analysis_emitted = False

        try:
            for event in main_graph.stream(
                {"task": request.task},
                config=config,
                stream_mode=["messages", "values"],
                subgraphs=True,
            ):
                # mode = event[0]
                # data = event[1]
                namespace, mode, data = event

                # values mode: emit metadata once we have routing info
                if mode == "values":
                    state = data
                    if (
                        not task_analysis_emitted
                        and state.get("recommended_architecture") is not None
                        and state.get("confidence") is not None
                    ):
                        task_analysis_emitted = True
                        yield _sse({
                            "type":       "metadata",
                            "routing":    state.get("recommended_architecture"),
                            "task_type":  state.get("task_type"),
                            "confidence": state.get("confidence"),
                        })
                        await asyncio.sleep(0)

                # messages mode: stream tokens from agent nodes
                elif mode == "messages":
                    chunk, meta = data
                    node = meta.get("langgraph_node", "")

                    yield _sse({"type": "node_change", "node": node})

                    if node in ("single_agent_node", "synthesizer"):
                        if hasattr(chunk, "content") and chunk.content:
                            yield _sse({
                                "type":    "token",
                                "content": chunk.content,
                                "node":    node,
                            })
                            await asyncio.sleep(0)

            # run complete
            snapshot = main_graph.get_state(config=config)
            shadow_run(snapshot)
            append_to_json("run_history.json", dict(snapshot.values))

            state = snapshot.values
            yield _sse({
                "type":   "done",
                "run_id": state.get("run_id", ""),
            })

        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
        },
    )


@app.get("/metrics")
def get_metrics():
    """
    Aggregate metrics over all runs in run_history.json.

    Returns:
      {
        "total_runs": int,
        "avg_latency_ms":  {"single": float, "multi": float},
        "avg_judge_score": {"single": float, "multi": float},
        "routing_distribution": {"single": int, "multi": int, ...}
      }
    """
    runs = _load_history()

    if not runs:
        return {
            "total_runs": 0,
            "avg_latency_ms": {},
            "avg_judge_score": {},
            "routing_distribution": {},
        }

    latencies:    dict[str, list[float]] = defaultdict(list)
    judge_scores: dict[str, list[int]]   = defaultdict(list)
    routing_dist: dict[str, int]         = defaultdict(int)

    for run in runs:
        archs = run.get("architectures_run") or []
        arch  = archs[0] if len(archs) == 1 else ("both" if archs else "unknown")

        routing_dist[arch] += 1

        lat = run.get("latency_ms")
        if lat is not None:
            latencies[arch].append(float(lat))

        if arch == "single" and run.get("judge_score_single") is not None:
            judge_scores["single"].append(run["judge_score_single"])
        elif arch == "multi" and run.get("judge_score_multi") is not None:
            judge_scores["multi"].append(run["judge_score_multi"])
        elif arch == "both":
            if run.get("judge_score_single") is not None:
                judge_scores["single"].append(run["judge_score_single"])
            if run.get("judge_score_multi") is not None:
                judge_scores["multi"].append(run["judge_score_multi"])

    avg_latency   = {k: round(statistics.mean(v), 1) for k, v in latencies.items()    if v}
    avg_judge     = {k: round(statistics.mean(v), 1) for k, v in judge_scores.items() if v}

    return {
        "total_runs":           len(runs),
        "avg_latency_ms":       avg_latency,
        "avg_judge_score":      avg_judge,
        "routing_distribution": dict(routing_dist),
    }


@app.get("/history")
def get_history(limit: int = 50):
    """
    Return the last `limit` runs from run_history.json, oldest-first.
    The Dashboard reverses the list client-side for newest-first display.
    """
    runs = _load_history()
    trimmed = runs[-limit:]

    result = []
    for run in trimmed:
        archs = run.get("architectures_run") or []
        arch  = archs[0] if len(archs) == 1 else ("both" if archs else None)

        # Derive a single judge_score field the Dashboard's HistoryRow expects
        if arch == "single":
            judge_score = run.get("judge_score_single")
        elif arch == "multi":
            judge_score = run.get("judge_score_multi")
        else:
            best = max(
                filter(None, [run.get("judge_score_single"), run.get("judge_score_multi")]),
                default=None,
            )
            judge_score = best

        result.append({
            "run_id":           run.get("run_id"),
            "task":             run.get("task"),
            "architectures_run": archs,
            "task_type":        run.get("task_type"),
            "confidence":       run.get("confidence"),
            "judge_score":      judge_score,
            "latency_ms":       round(run["latency_ms"], 1) if run.get("latency_ms") else None,
            "success":          run.get("success"),
            "was_routing_correct": run.get("was_routing_correct"),
        })

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)