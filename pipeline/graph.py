import uuid
import time
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from models.state import TaskState
from agents.task_analysis import task_analysis_node
from agents.single_agent import single_agent_subgraph
from agents.multi_agent import multi_agent_subgraph
from agents.judge import judge_node
from core.router import confidence_router
from core.decision import optimal_decision


checkpointer = MemorySaver()


def input_node(state: TaskState):
    task_id = str(uuid.uuid4())
    start_time = time.time()
    return {
        'task': state['task'],
        'run_id': task_id,
        'start_time': start_time,
        'task_type': None,
        'taxonomy_reasoning': None,
        'raw_confidence': None,
        'confidence': None,
        'recommended_architecture': None,
        'complexity_features': None,
        'profiler_reasoning': None,
        'routing_history': None,
        'architectures_run': [],
        'single_output': None,
        'multi_output': None,
        'final_output': None,
        'judge_success_score': None,
        'judge_architecture_fit': None,
        'judge_explanation': None,
        'judge_score_single': None,
        'judge_score_multi': None,
        'judge_winner': None,
        'score_differential': None,
        'was_routing_correct': None,
        'optimal_solution_architecture': None,
        'decision_basis': None,
        'optimal_output': None,
        'latency_ms': None,
        'single_tokens': None,
        'multi_tokens': None,
        'total_tokens': None,
        'llm_call_count': None,
        'success': None,
    }


def output_node(state: TaskState) -> dict:
    both_ran = set(state['architectures_run']) == {'single', 'multi'}
    architecture_used = 'both' if both_ran else state['architectures_run'][0]

    print("\n" + "="*60)
    print(f"TASK: {state['task']}")
    print("="*60)
    print(f"Architecture Used:        {architecture_used}")
    print(f"Recommended:              {state['recommended_architecture']}")
    print(f"Confidence:               {state['confidence']:.2f}")
    print(f"Routing Correct:          {state['was_routing_correct']}")
    print(f"Profiler Reasoning:       {state['profiler_reasoning']}")
    print("-"*60)
    print(f"Judge Score (Single):     {state['judge_score_single']}")
    print(f"Judge Score (Multi):      {state['judge_score_multi']}")
    print(f"Judge Winner:             {state['judge_winner']}")
    print(f"Score Differential:       {state['score_differential']}")
    print(f"Judge Architecture Fit:   {state['judge_architecture_fit']}")
    print(f"Judge Explanation:        {state['judge_explanation']}")
    print("-"*60)
    print(f"Optimal Architecture:     {state['optimal_solution_architecture']}")
    print(f"Decision Basis:           {state['decision_basis']}")
    print(f"Single Tokens:            {state['single_tokens']}")
    print(f"Multi Tokens:             {state['multi_tokens']}")
    print(f"Total Tokens:             {state['total_tokens']}")
    print(f"Latency (ms):             {state['latency_ms']:.2f}")
    print(f"LLM Call Count:           {state['llm_call_count']}")
    print("="*60)
    print("OPTIMAL OUTPUT:")
    print(state['optimal_output'])
    print("="*60 + "\n")

    return {}


builder = StateGraph(TaskState)
builder.add_node('Input', input_node)
builder.add_node('Task Analysis', task_analysis_node)
builder.add_node('Single Agent', single_agent_subgraph, return_only_outputs=True)
builder.add_node('Multi Agent', multi_agent_subgraph, return_only_outputs=True)
builder.add_node('Judge', judge_node)
builder.add_node('Optional Decision Maker', optimal_decision)
builder.add_node('Output', output_node)

builder.add_edge(START, 'Input')
builder.add_edge('Input', 'Task Analysis')
builder.add_conditional_edges(
    'Task Analysis',
    confidence_router,
    {
        'single': 'Single Agent',
        'multi': 'Multi Agent'
    }
)
builder.add_edge('Single Agent', 'Judge')
builder.add_edge('Multi Agent', 'Judge')
builder.add_edge('Judge', 'Optional Decision Maker')
builder.add_edge('Optional Decision Maker', 'Output')
builder.add_edge('Output', END)

main_graph = builder.compile(checkpointer=checkpointer)
