from typing import Literal
from pydantic import Field, BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import TaskState


class Judge(BaseModel):
    score: int = Field(
        ...,
        description="Score out of 100 for how well the response addressed the task"
    )
    architecture_fit: Literal['correct', 'suboptimal', 'wrong'] = Field(
        ...,
        description="Whether the architecture used was the right choice for this task"
    )
    explanation: str = Field(
        ...,
        description="Short explanation of the score and architecture assessment"
    )


judge_llm = ChatOllama(
    model='llama3.1',
    temperature=0.1,
    num_ctx=8192,
    num_predict=512,
    repeat_penalty=1.1,
).with_structured_output(Judge)

JUDGE_SYSTEM_PROMPT = """
You are an evaluation judge for an AI orchestration system.

You will receive the original task, the architecture used, and the response produced.

Score (0-100):
- 90-100: Complete, accurate, well structured
- 70-89: Mostly complete with minor gaps  
- 50-69: Partially addresses the task
- Below 50: Significant gaps or errors

Architecture fit:
- correct: Right architecture for this task type
- suboptimal: Other architecture would have been more efficient
- wrong: Other architecture would have produced meaningfully better output

Be consistent — your scores feed into a research benchmark."""


def judge_node(state: TaskState):

    both_ran = set(state['architectures_run']) == {'single', 'multi'}
    architecture_used = 'both' if both_ran else state['architectures_run'][0]

    if both_ran:
        # Judge evaluates both outputs separately and scores each
        single_prompt = (
            f"Task: {state['task']}\n\n"
            f"Response:\n{state['single_output']}\n\n"
            "Score this response out of 100 based on the task given."
        )

        multi_prompt = (
            f"Task: {state['task']}\n\n"
            f"Response:\n{state['multi_output']}\n\n"
            "Score this response out of 100 based on the task given."
        )

        for _ in range(2):
            try:
                single_decision = judge_llm.invoke([
                    SystemMessage(JUDGE_SYSTEM_PROMPT),
                    HumanMessage(single_prompt)
                ])
                multi_decision = judge_llm.invoke([
                    SystemMessage(JUDGE_SYSTEM_PROMPT),
                    HumanMessage(multi_prompt)
                ])

                score_differential = multi_decision.score - single_decision.score
                judge_winner = 'multi' if score_differential > 0 else 'single'
                was_routing_correct = (judge_winner == state['recommended_architecture'])
                judge_decision = multi_decision if score_differential > 0 else single_decision

                final_output = (state['multi_output'] if judge_winner == 'multi' else state['single_output'])

                return {
                    'judge_score_single': single_decision.score,
                    'judge_score_multi': multi_decision.score,
                    'judge_winner': judge_winner,
                    'score_differential': score_differential,
                    'was_routing_correct': was_routing_correct,
                    'judge_success_score': max(single_decision.score, multi_decision.score),
                    'judge_architecture_fit': judge_decision.architecture_fit,
                    'judge_explanation': judge_decision.explanation,
                    'final_output': final_output,
                    'success': max(single_decision.score, multi_decision.score) >= 70,
                    'total_tokens': state['multi_tokens'] if judge_winner == 'multi' else state['single_tokens']
                }
            except Exception as e:
                print(f"Judge Node error: {type(e).__name__}: {e}")
                continue

        return {
            'judge_score_single': None,
            'judge_score_multi': None,
            'judge_winner': None,
            'score_differential': None,
            'routing_was_correct': None,
            'judge_success_score': None,
            'judge_architecture_fit': None,
            'judge_explanation': None,
            'final_output': None,
            'success': None
        }

    else:
        prompt = (
            f"Task: {state['task']}\n"
            f"Architecture used: {architecture_used}\n"
            f"Response:\n{state['final_output']}\n\n"
            "Score this response out of 100 based on the task given and assess architecture fit."
        )

        for _ in range(2):
            try:
                decision = judge_llm.invoke([
                    SystemMessage(JUDGE_SYSTEM_PROMPT),
                    HumanMessage(prompt)
                ])
                return {
                    'judge_score_single': decision.score if architecture_used == 'single' else None,
                    'judge_score_multi': decision.score if architecture_used == 'multi' else None,
                    'judge_winner': architecture_used,
                    'score_differential': None,
                    'was_routing_correct': True,
                    'judge_success_score': decision.score,
                    'judge_architecture_fit': decision.architecture_fit,
                    'judge_explanation': decision.explanation,
                    'success': decision.score >= 70,
                    'final_output': state['final_output']
                }
            except Exception as e:
                print(f"Judge Node error: {type(e).__name__}: {e}")
                continue

        return {
            'judge_score_single': None,
            'judge_score_multi': None,
            'judge_winner': None,
            'score_differential': None,
            'routing_was_correct': None,
            'judge_success_score': None,
            'judge_architecture_fit': None,
            'judge_explanation': None,
            'final_output': None,
            'success': None
        }
