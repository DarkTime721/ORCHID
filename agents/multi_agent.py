import time
import operator
from typing import Annotated

from pydantic import Field, BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Send

from models.state import keep_last
from core.utils import encoder
from agents.single_agent import search_tool, calculator


class Section(BaseModel):
    name: str = Field(..., description='The section name')
    description: str = Field(..., description='Short description of the subtask')


class Sections(BaseModel):
    sections: list[Section] = Field(..., description='List of subtasks')


class MultiAgentState(MessagesState):
    task: Annotated[str, keep_last]
    subtasks: list[Section]
    agent_outputs: Annotated[list[str], operator.add]
    formatter_dict: Annotated[list[dict], operator.add]
    multi_output: Annotated[str, keep_last]
    final_output: Annotated[str, keep_last]
    llm_call_count: Annotated[int, keep_last]
    total_tokens: Annotated[int, keep_last]
    latency_ms: Annotated[float, keep_last]
    subgraph_start_time: Annotated[float, keep_last]
    multi_tokens: Annotated[int, keep_last]
    architectures_run: Annotated[list[str], operator.add]


class WorkerState(MessagesState):
    subtask: Section
    agent_outputs: Annotated[list[str], operator.add]
    tokens: Annotated[int, keep_last]
    llm_calls: Annotated[int, keep_last]
    formatter_dict: Annotated[list[dict], operator.add]


ORCHESTRATOR_SYSTEM = """You are a task decomposition system.
Divide the task into 2-5 clear independent subtasks.
Each subtask should be completable without needing other subtasks' results."""

WORKER_SYSTEM = """
You are a specialized research agent. 
For your assigned subtask:
- Provide a comprehensive, detailed response
- Include relevant examples, data, and explanations
- Use tools when current information is needed
- Structure your response clearly with key points
"""

SYNTHESIZER_PROMPT = """
You are a synthesis agent. You will receive multiple research outputs from specialized agents.
Your job is to combine them into a single coherent, well-structured response that:
- Flows naturally as one unified answer
- Eliminates redundancy
- Preserves all key information
- Has clear structure and transitions between topics
"""

multi_agent_orchestrator = ChatOllama(model='qwen3:8b', temperature=0.6, top_p=0.95, top_k=20).with_structured_output(Sections)
multi_agent_worker = ChatOllama(model='qwen3:8b', temperature=0.6, top_p=0.95, top_k=20).bind_tools([search_tool, calculator])
multi_synthesizer_llm = ChatOllama(model='qwen3:8b', temperature=0.6, top_p=0.95, top_k=20)


def subgraph_entry(state: MultiAgentState) -> dict:
    return {'subgraph_start_time': time.time(),
            'architectures_run': ['multi']}


def orchestrator_node(state: MultiAgentState) -> dict:
    result = multi_agent_orchestrator.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=f"Task: {state['task']}")
    ])
    return {'subtasks': result.sections}


def assign_workers(state: MultiAgentState) -> list[Send]:
    return [
        Send("worker_node", {"subtask": section, "messages": []})
        for section in state["subtasks"]
    ]


def worker_agent(state: WorkerState) -> dict:
    messages = state['messages']
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [
            SystemMessage(WORKER_SYSTEM)
        ] + messages
    response = multi_agent_worker.invoke(messages)
    return {'agent_outputs': [response.content],
            'messages': [response]
            }


def worker_entry(state: WorkerState) -> dict:
    return {
        'messages': [
            HumanMessage(content=f"Subtask: {state['subtask'].name}\nDescription: {state['subtask'].description}")
        ]
    }


def worker_output_formatter(state: WorkerState) -> dict:
    last_msg = state['messages'][-1]
    tokens = len(encoder.encode(last_msg.content))
    return {
        'formatter_dict': [{
            'tokens': tokens,
            'llm_calls': len(state['agent_outputs']),
            'completed': True
        }]
    }


def synthesizer(state: MultiAgentState) -> dict:
    agent_outputs = state['agent_outputs']

    combined = "\n\n---\n\n".join(
        agent_outputs
    )

    synthesized_output = multi_synthesizer_llm.invoke([
        SystemMessage(SYNTHESIZER_PROMPT),
        HumanMessage(content=f"Task: {state['task']}\n\nAgent outputs:\n{combined}")
    ])
    return {
        'final_output': synthesized_output,
        'multi_output': synthesized_output
    }


def output_formatter(state: MultiAgentState) -> dict:
    latency = (time.time() - state['subgraph_start_time']) * 1000

    orchestrator_tokens = len(encoder.encode(state['task']))

    total_tokens = sum(
        item['tokens']
        for item in state['formatter_dict']
        if item['completed']
    ) + orchestrator_tokens

    return {
        # 'final_output': state['final_output'],
        # 'multi_output': state['final_output'],
        'latency_ms': latency,
        'llm_call_count': sum(item.get('llm_calls', 1) for item in state['formatter_dict']) + 2,   # worker llms + orchestrator llm + synthesizer llm
        'multi_tokens': total_tokens,
        'total_tokens': total_tokens,
    }


worker_graph = StateGraph(WorkerState)
worker_graph.add_node('worker_entry', worker_entry)
worker_graph.add_node('worker_agent', worker_agent)
worker_graph.add_node('tools', ToolNode([search_tool, calculator]))
worker_graph.add_node('worker_output_formatter', worker_output_formatter)

worker_graph.add_edge(START, 'worker_entry')
worker_graph.add_edge('worker_entry', 'worker_agent')
worker_graph.add_conditional_edges(
    'worker_agent',
    tools_condition,
    {
        'tools': 'tools',
        END: 'worker_output_formatter'
    }
)

worker_graph.add_edge('tools', 'worker_agent')
worker_graph.add_edge('worker_output_formatter', END)

worker_subgraph = worker_graph.compile()

multi_agent_graph = StateGraph(MultiAgentState)

multi_agent_graph.add_node('subgraph_entry', subgraph_entry)
multi_agent_graph.add_node('orchestrator_node', orchestrator_node)
multi_agent_graph.add_node('worker_node', worker_subgraph)
multi_agent_graph.add_node('synthesizer', synthesizer)
multi_agent_graph.add_node('output_formatter', output_formatter)

multi_agent_graph.add_edge(START, 'subgraph_entry')
multi_agent_graph.add_edge('subgraph_entry', 'orchestrator_node')
multi_agent_graph.add_conditional_edges(
    'orchestrator_node',
    assign_workers,
    ['worker_node']
)
multi_agent_graph.add_edge('worker_node', 'synthesizer')
multi_agent_graph.add_edge('synthesizer', 'output_formatter')
multi_agent_graph.add_edge('output_formatter', END)

multi_agent_subgraph = multi_agent_graph.compile()
