from typing import Literal
from pydantic import Field, BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from models.state import TaskState
from core.history import load_routing_history
from core.confidence import compute_multi_scores, compute_confidence


class TaskAnalysis(BaseModel):
    task_type: Literal[
        'decomposable', 'sequential',
        'convergent', 'parallel', 'ambiguous', 'atomic'
    ] = Field(..., description="Chosen taxonomy type for the task.")
    taxonomy_reasoning: str = Field(..., description="Reasoning for why the taxonomy type was chosen for the task.")
    subtask_count: int = Field(..., description="The count of subtasks in which the given task can be divided into.")
    domain_breadth: int = Field(..., description="The breadth of the domain which the task spans.")
    parallelism_potential: bool = Field(..., description="Gives the information about the fact that if the task can be parallized or not.")
    interdependency: Literal['low', 'medium', 'high'] = Field(..., description="Gives the interdependency of the task")
    context_volume: Literal['low', 'medium', 'high'] = Field(..., description="Gives the context volume required for the task.")
    recommended_architecture: Literal['single', 'multi'] = Field(..., description="Recommend architecture suitable for the given task")
    task_complexity: Literal['trivial', 'simple', 'moderate', 'complex'] = Field(..., description="Overall task complexity")
    profiler_reasoning: str = Field(..., description="Reasoning for the architecture recommendation")


task_analysis_llm = ChatOllama(model='qwen2.5', temperature=0.0, top_p=1.0, num_predict=256).with_structured_output(TaskAnalysis)

TASK_ANALYSIS_PROMPT = """
You are a task analysis system for an AI orchestration pipeline that routes tasks
to either a single-agent or multi-agent architecture.

Analyze the given task and return a structured analysis covering taxonomy, complexity
features, and a routing recommendation. You do NOT output a confidence score —
confidence is computed externally from your feature outputs.

---

TAXONOMY - Classify the task into one of these types:
- atomic: single, indivisible task with no meaningful decomposition possible
  (single factual lookups, simple calculations, basic definitions)
- decomposable: task naturally breaks into independent subtasks that can be solved separately
- sequential: subtasks exist but each depends on the previous result (pipeline-like)
- convergent: multiple information streams that must be gathered and synthesized into one answer
- parallel: multiple truly independent subtasks with no dependency between them
- ambiguous: use when the task scope or requirements are genuinely unclear and the task
    cannot be confidently placed into any other taxonomy type. Specifically:
    - the task could plausibly be atomic, decomposable, or parallel depending on
      unstated assumptions about scope
    - the user's intent is underspecified and different interpretations would lead to
      fundamentally different decompositions
    - it is unclear whether subtasks even exist, not just how many there are
    Do NOT use ambiguous just because a task is complex, cross-domain, or hard to route.
    Ambiguous is about unclear scope, not unclear routing.

    Only choose a taxonomy type if the structure is clearly supported by the task.
    If identifying a structure requires assumptions that are not explicitly stated,
    classify the task as ambiguous instead.

    When multiple plausible structures exist due to missing scope or constraints,
    prefer ambiguous over forcing a dominant structure.
    

STRICT AMBIGUITY RULE:
- If the task uses vague terms such as "insights", "analysis", "overview",
  "tell me about", or similar without specifying scope, entities, or constraints,
  classify it as ambiguous.

- Do NOT assume subtasks, decomposition, or parallelism unless clearly implied.
  The goal is to detect real structure, not to construct a plausible one.

- If the task allows multiple valid interpretations that would lead to different
  task structures (e.g., atomic vs decomposable, single vs parallel),
  classify it as ambiguous.

- This includes conditional, optional, or alternative phrasing (e.g., "or",
  "if needed", "depending on"), but is NOT limited to specific keywords.
  Focus on whether the task defines multiple possible execution paths.

- If different reasonable interpretations would change subtask_count,
  parallelism_potential, or interdependency, the task is ambiguous.


TAXONOMY RULES:
- atomic is for tasks with no meaningful structure — do not use it just because a task is simple
  but has identifiable subtasks
- ambiguous is for tasks where the scope itself is unclear, not for tasks that are merely complex
- parallel requires subtasks that are genuinely independent with zero dependency until aggregation
- convergent is for tasks where multiple streams must be gathered but synthesis is the primary goal
- when in doubt between two taxonomy types, choose the one that best describes the dominant
  structure of the task, and explain the ambiguity in taxonomy_reasoning

---

COMPLEXITY FEATURES:
- subtask_count: how many meaningful subtasks the task decomposes into
  (be realistic and conservative — do not inflate this number)
- domain_breadth: number of distinct knowledge domains involved, scored 1-5
  (1 = single domain, 5 = highly cross-domain. Be precise — finance and economics
  are not two separate domains, but medicine and law are)
- parallelism_potential: true if subtasks can be executed independently 
  for a significant portion of their work, even if a final aggregation step is required
- interdependency: how tightly subtasks depend on each other
  - low: subtasks are fully independent, outputs do not feed into each other
  - medium: some dependency exists but subtasks are mostly separable
  - high: subtasks are tightly coupled, strict order matters, earlier outputs feed later ones
- context_volume: how much background knowledge or external information is required
  - low: self-contained, minimal context needed
  - medium: moderate domain knowledge required
  - high: deep expertise or large amounts of external information needed
- task_complexity: overall difficulty of the task
  - trivial: single fact, basic definition, simple calculation
  - simple: straightforward explanation or analysis of one concept
  - moderate: multi-part analysis or comparison requiring some depth
  - complex: deep research, cross-domain synthesis, or comparison across many entities
- parallelism_potential should be TRUE for:
  - tasks involving multiple entities (e.g., multiple companies, cities, datasets)
  - tasks where the same operation is applied across multiple items
  - tasks where data gathering can happen independently before final synthesis

FEATURE CONSISTENCY RULES — these combinations are contradictory and must not co-occur:
- parallelism_potential=true AND interdependency=high: parallelism requires independence,
  high interdependency means tasks cannot run in parallel — pick one
- subtask_count=1 AND parallelism_potential=true: a single task cannot be parallelized
- task_complexity=trivial AND subtask_count > 2: trivial tasks do not decompose meaningfully

---

CONSISTENCY RULE:
- The task_type must be consistent with taxonomy_reasoning.

- If your reasoning indicates that the task is ambiguous, has unclear scope,
  or allows multiple valid interpretations leading to different structures,
  you MUST set task_type to "ambiguous".

- Do NOT describe ambiguity in taxonomy_reasoning while assigning a non-ambiguous task_type.

- When in doubt, prefer internal consistency over forcing a specific taxonomy label.

---

ROUTING DECISION:
- recommended_architecture:
  - single: for trivial/simple tasks, sequential tasks, tasks with high interdependency,
    or any task where parallelism provides no meaningful benefit even if subtasks exist
  - multi: when the task meets ANY of the following sufficient conditions:
      CONDITION A (parallel structure): subtask_count >= 3 AND parallelism_potential is true
        AND interdependency is low AND task_complexity is moderate or complex
      CONDITION B (cross-domain): domain_breadth >= 2 AND interdependency is low
        AND task_complexity is moderate or complex
      CONDITION C (high volume parallel): context_volume is high AND parallelism_potential
        is true AND subtask_count >= 2
    In all cases, multi is only valid if orchestration overhead is justified — do not recommend
    multi for tasks a single agent can handle end-to-end without meaningful quality loss

- profiler_reasoning: concise explanation of why you chose this architecture, explicitly
  stating which CONDITION (A, B, or C) was met for multi, or why none were met for single.
  Reference specific feature values — subtask_count, interdependency, domain_breadth,
  parallelism_potential — by name and value in your reasoning. Be precise, not general.
  If the task almost but not quite met a condition, state which condition and which
  feature fell short.
  If recommending ambiguous, explicitly state which conditions were partially met,
  which features were contradictory or unclear, and what information would resolve
  the ambiguity.

---

IMPORTANT GUIDELINES:
- Do NOT recommend multi just because subtask_count > 1. Many tasks decompose into subtasks
  but are still better handled by a single agent.
- Sequential tasks usually favour single due to dependency chains.

Convergent tasks MAY benefit from multi-agent architectures if:
- subtask_count >= 3 AND
- subtasks involve different entities or data sources AND
- a significant portion of work can be done independently before synthesis

Do NOT assume convergent tasks are single by default — evaluate parallelism_potential independently.
- Trivial and simple tasks should always be single regardless of other features.
- A task being single-domain (domain_breadth=1) does NOT disqualify it from multi — if it
  has 3+ genuinely parallel subtasks of moderate/complex difficulty, CONDITION A still applies.
- Only recommend multi when specialized parallel agents produce meaningfully better output
  than a single agent handling the task end-to-end.
- Ambiguous taxonomy is not a failure — it is an honest signal. Classify as ambiguous when
  task scope is genuinely unclear, not just when the task is complex or multi-part.
- Your feature outputs directly determine confidence in downstream systems — accuracy of
  subtask_count, interdependency, and parallelism_potential is more important than the
  architecture recommendation itself. Be precise and conservative on these fields.



Task: """


def task_analysis_node(state: TaskState):
    routing_history = None

    for _ in range(3):
        try:
            result = task_analysis_llm.invoke([
                SystemMessage(TASK_ANALYSIS_PROMPT),
                HumanMessage(content=f"Task: {state['task']}")
            ])

            routing_history = load_routing_history(task_type=result.task_type)
            multi_scores = compute_multi_scores(result)
            adjustment = routing_history['recommended_adjustment'] if routing_history else 0.0

            raw_confidence = compute_confidence(result=result, multi_scores=multi_scores, adjustment=0.0)
            confidence = compute_confidence(result=result, multi_scores=multi_scores, adjustment=adjustment)

            return {
                'task_type': result.task_type,
                'taxonomy_reasoning': result.taxonomy_reasoning,
                'complexity_features': {
                    'subtask_count': result.subtask_count,
                    'domain_breadth': result.domain_breadth,
                    'parallelism_potential': result.parallelism_potential,
                    'interdependency': result.interdependency,
                    'context_volume': result.context_volume,
                    'task_complexity': result.task_complexity,
                },
                'recommended_architecture': result.recommended_architecture,
                'raw_confidence': raw_confidence,
                'confidence': confidence,
                'profiler_reasoning': result.profiler_reasoning,
                'routing_history': routing_history
            }
        except Exception as e:
            print(f"Task analysis failed: {e}")
            continue

    return {
        'task_type': 'straightforward',
        'taxonomy_reasoning': 'Analysis failed',
        'complexity_features': None,
        'recommended_architecture': 'single',
        'raw_confidence': None,
        'confidence': None,
        'profiler_reasoning': 'Task analysis failed after 3 attempts',
        'routing_history': routing_history
    }
