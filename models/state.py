from typing import TypedDict, Literal, Optional, Annotated
import operator


def keep_last(a, b):
    return b


class RoutingHistorySummary(TypedDict):
    task_type: str
    total_runs: int
    single_success_rate: float
    multi_success_rate: float
    recommended_adjustment: float


class ComplexityFeatures(TypedDict):
    subtask_count: int
    domain_breadth: int
    parallelism_potential: bool
    interdependency: Literal['low', 'medium', 'high']
    context_volume: Literal['low', 'medium', 'high']
    task_complexity: Literal['triival', 'simple', 'moderate', 'complex']


class TaskState(TypedDict):
    # Input required
    task: Annotated[str, keep_last]
    run_id: str
    start_time: float

    # Task Analysis Layer
    task_type: Optional[Literal['decomposable', 'sequential', 'convergent', 'parallel', 'ambiguous', 'atomic']]
    taxonomy_reasoning: Optional[str]
    raw_confidence: Optional[float]
    confidence: Optional[float]
    recommended_architecture: Optional[Literal['single', 'multi']]
    complexity_features: Optional[ComplexityFeatures]
    profiler_reasoning: Optional[str]

    # Feedback Loop
    routing_history: Optional[RoutingHistorySummary]

    # Execution
    architectures_run: Annotated[list[str], operator.add]
    single_output: Annotated[Optional[str], keep_last]
    multi_output: Annotated[Optional[str], keep_last]
    final_output: Annotated[Optional[str], keep_last]

    # Judge
    judge_success_score: Optional[int]
    judge_architecture_fit: Optional[Literal['correct', 'suboptimal', 'wrong']]
    judge_explanation: Optional[str]
    judge_score_single: Optional[int]
    judge_score_multi: Optional[int]
    judge_winner: Optional[Literal['single', 'multi']]
    score_differential: Optional[float]
    was_routing_correct: Optional[bool]

    # Optimal solution (based on tokens consumed and scores given by judge node)
    optimal_solution_architecture: Optional[Literal['single', 'multi']]
    decision_basis: Optional[Literal['cost', 'quality', 'single_run']]
    optimal_output: Annotated[Optional[str], keep_last]

    # Metrics
    single_tokens: Annotated[Optional[int], keep_last]
    multi_tokens: Annotated[Optional[int], keep_last]
    total_tokens: Annotated[Optional[int], keep_last]
    llm_call_count: Annotated[Optional[int], keep_last]
    latency_ms: Annotated[Optional[float], keep_last]
    success: Annotated[Optional[bool], keep_last]
