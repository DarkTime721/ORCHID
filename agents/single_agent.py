import time
import operator
from typing import Optional, Annotated

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from models.state import keep_last
from core.utils import count_message_tokens

MAX_ITERATIONS = 5


class SingleAgentState(MessagesState):
    task: str
    single_output: str
    final_output: str
    llm_call_count: int
    total_tokens: int
    latency_ms: float
    subgraph_start_time: float
    architectures_run: Annotated[list[str], operator.add]
    single_tokens: Annotated[Optional[int], keep_last]


SYSTEM_PROMPT = """
You are a capable research and reasoning assistant.

For each task:
- Use search_tool to find current, factual information when needed
- Use calculator for any mathematical computations
- Always verify numerical claims with calculator rather than estimating
- Provide a comprehensive, well-structured response
- Cite what you found via search when relevant

Output must be complete, accurate, and directly address the task.
"""

search_tool = DuckDuckGoSearchRun()
# python_tool = PythonREPLTool()
# Cannot add the PythonREPL tool due to langchain's unresolved import change which breaks the compatibility with 1.x versions


@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Input should be a valid Python math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


def subgraph_entry(state: SingleAgentState) -> dict:
    return {
        'subgraph_start_time': time.time(),
        'architectures_run': ['single'],
        'messages': [HumanMessage(content=state['task'])]
    }


single_agent_llm = ChatOllama(model='qwen3:8b', temperature=0.6, top_p=0.95, top_k=20)
single_agent_llm = single_agent_llm.bind_tools([search_tool, calculator])


def single_agent(state: SingleAgentState):
    ai_msg = [m for m in state['messages'] if hasattr(m, 'type') and m.type == 'ai']

    if len(ai_msg) >= MAX_ITERATIONS:
        final_prompt = state['messages'] + [
            SystemMessage(
                "You have reached the maximum number of steps. "
                "Provide your final answer now based on what you have gathered so far."
            )
        ]
        response = single_agent_llm.invoke(final_prompt)
        return {
            'messages': [response]
        }

    if not any(isinstance(m, SystemMessage) for m in state['messages']):
        messages = [SYSTEM_PROMPT] + state['messages']
    response = single_agent_llm.invoke(messages)
    return {
        'messages': [response]
    }


def output_formatter(state: SingleAgentState) -> dict:
    latency = (time.time() - state['subgraph_start_time']) * 1000
    ai_msg = [
        m for m in state['messages'] if hasattr(m, 'type') and m.type == 'ai'
    ]

    last_msg = state['messages'][-1]
    total_tokens = count_message_tokens(state['messages'])

    return {
        'final_output': last_msg.content,
        'single_output': last_msg.content,
        'latency_ms': latency,
        'llm_call_count': len(ai_msg),
        'single_tokens': total_tokens,
        'total_tokens': total_tokens
    }


single_agent_graph = StateGraph(SingleAgentState)

single_agent_graph.add_node('subgraph_entry', subgraph_entry)
single_agent_graph.add_node('single_agent_node', single_agent)
single_agent_graph.add_node('tools', ToolNode([search_tool, calculator]))
single_agent_graph.add_node('output_formatter', output_formatter)

single_agent_graph.add_edge(START, 'subgraph_entry')
single_agent_graph.add_edge('subgraph_entry', 'single_agent_node')
single_agent_graph.add_conditional_edges(
    'single_agent_node',
    tools_condition,
    {
        'tools': 'tools',
        END: 'output_formatter'
    }
)
single_agent_graph.add_edge('tools', 'single_agent_node')
single_agent_graph.add_edge('output_formatter', END)

single_agent_subgraph = single_agent_graph.compile()
