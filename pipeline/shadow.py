import uuid
import random

from agents.single_agent import single_agent_subgraph
from agents.multi_agent import multi_agent_subgraph
from core.utils import append_to_json


def shadow_evaluation(snapshot):
    state = snapshot.values
    both_ran = set(state['architectures_run']) == {'single', 'multi'}
    if not both_ran:

        shadow_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        subgraph_state = {}

        if state['architectures_run'][0] == 'single':
            for mode, data in multi_agent_subgraph.stream(
                input={'task': state['task']},
                config=state,
                stream_mode=['messages', 'values']
            ):
                if mode == 'messages':
                    chunk, metadata = data
                    node = metadata.get('langgraph_node', '')

                    if node == 'synthesizer' and hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end="", flush=True)

                elif mode == 'values':
                    subgraph_state = data
            append_to_json('shadow_run.json', dict(subgraph_state))

        else:
            for mode, data in single_agent_subgraph.stream(
                input={'task': state['task']},
                config=shadow_config,
                stream_mode=['messages', 'values']
            ):
                if mode == 'messages':
                    chunk, metadata = data
                    node = metadata.get('langgraph_node', '')

                    if node == 'synthesizer' and hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end="", flush=True)

                elif mode == 'values':
                    subgraph_state = data
            append_to_json('shadow_run.json', dict(subgraph_state))
    else:
        return


def shadow_run(snapshot):
    if random.random() < 0.1:
        shadow_evaluation(snapshot)
