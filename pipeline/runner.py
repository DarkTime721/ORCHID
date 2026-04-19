import uuid

from pipeline.graph import main_graph
from pipeline.shadow import shadow_run
from core.utils import append_to_json


def stream_pipeline(task: str):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    for event in main_graph.stream(
        {"task": task},
        config=config,
        stream_mode="messages",
        subgraphs=True
    ):
        chunk, metadata = event[1]
        node = metadata.get("langgraph_node", "")
        if node in ("single_agent_node", "synthesizer") and hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)

    snapshot = main_graph.get_state(config=config)
    shadow_run(snapshot)
    append_to_json('run_history.json', dict(snapshot.values))
