from models.state import TaskState

QUALITY_WEIGHT = 0.7
TOKEN_WEIGHT = 0.2
LATENCY_WEIGHT = 0.1
QUALITY_EQUIVALENCE_THRESHOLD = 15


def optimal_decision(state: TaskState) -> dict:

    both_ran = set(state['architectures_run']) == {'single', 'multi'}
    architecture_used = 'both' if both_ran else state['architectures_run'][0]

    if architecture_used != 'both':
        return {
            'optimal_solution_architecture': state['architectures_run'][0],
            'decision_basis': 'single_run',
            'optimal_output': state['final_output']
        }

    single_quality = state['judge_score_single']
    multi_quality = state['judge_score_multi']

    if single_quality is None or multi_quality is None:
        print(f"[optimal_decision] Judge scores missing: single={single_quality}, multi={multi_quality}")
        return {
            'optimal_solution_architecture': None,
            'optimal_solution': None,
            'decision_basis': None
        }

    score_differential_abs = abs(state['score_differential'])

    if score_differential_abs < QUALITY_EQUIVALENCE_THRESHOLD:
        winner = 'single'   # As single is always cheaper
        decision_basis = 'cost'
    else:
        single_token_score = max(0, 1 - (state['single_tokens'] / 2000))
        multi_token_score = max(0, 1 - (state['multi_tokens'] / 2000))

        single_composite = (QUALITY_WEIGHT * single_quality / 100 + TOKEN_WEIGHT * single_token_score)
        multi_composite = (QUALITY_WEIGHT * multi_quality / 100 + TOKEN_WEIGHT * multi_token_score)

        winner = 'multi' if multi_composite > single_composite else 'single'
        decision_basis = 'quality'

    return {
        'optimal_solution_architecture': winner,
        'decision_basis': decision_basis,
        'optimal_output': state['single_output'] if winner == 'single' else state['multi_output']
    }
