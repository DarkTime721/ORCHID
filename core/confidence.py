import math
from typing import Dict


def calibrate(score: float, sharpness: float = 6.0, center: float = 0.55) -> float:
    return 1 / (1 + math.exp(-sharpness * (score - center)))


def compute_single_score(result) -> float:
    simple_structured = (
        result.task_complexity in ('trivial', 'simple') and
        result.domain_breadth == 1
    )

    score = (
        0.30 * (result.task_complexity in ('trivial', 'simple')) +
        0.25 * (result.interdependency in ('medium', 'high')) +
        0.25 * (not result.parallelism_potential) +
        0.15 * (result.subtask_count <= 3) +
        0.10 * simple_structured
    )

    return round(score, 4)


def compute_multi_scores(result) -> Dict[str, float]:
    complexity_ok = result.task_complexity in ('moderate', 'complex')

    condition_a = (
        0.5 * (result.subtask_count >= 3) +
        0.3 * result.parallelism_potential +
        0.2 * (result.interdependency == 'low')
    ) if complexity_ok else 0.0

    condition_b = (
        0.6 * (result.domain_breadth >= 2) +
        0.4 * (result.interdependency == 'low')
    ) if complexity_ok else 0.0

    condition_c = (
        0.4 * (result.context_volume == 'high') +
        0.3 * result.parallelism_potential +
        0.3 * (result.subtask_count >= 2)
    )

    return {
        'condition_a': round(condition_a, 4),
        'condition_b': round(condition_b, 4),
        'condition_c': round(condition_c, 4),
    }


def aggregate_multi_score(multi_scores: Dict[str, float]) -> float:
    weights = {
        'condition_a': 0.45,
        'condition_b': 0.35,
        'condition_c': 0.20,
    }
    return round(sum(weights[k] * multi_scores[k] for k in weights), 4)


def compute_confidence(result, multi_scores: Dict[str, float], adjustment: float = 0.0) -> float:

    if not (-0.2 <= adjustment <= 0.2):
        raise ValueError(f"adjustment must be in [-0.2, 0.2], got {adjustment}")

    single_score = compute_single_score(result)
    multi_score = aggregate_multi_score(multi_scores)

    single_cal = calibrate(single_score)
    multi_cal = calibrate(multi_score)

    winner_score = max(single_cal, multi_cal)
    loser_score = min(single_cal, multi_cal)

    ratio = loser_score / (winner_score + 1e-9)
    confidence = winner_score * (1.0 - 0.5 * (ratio ** 2))

    margin = abs(single_cal - multi_cal)

    if winner_score < 0.85:
        if margin > 0.4:
            confidence += 0.08
        elif margin > 0.25:
            confidence += 0.04

    if result.parallelism_potential and result.interdependency in ('medium', 'high'):
        confidence -= 0.07
    if result.recommended_architecture == 'single' and multi_score > 0.60:
        confidence -= 0.10
    elif result.recommended_architecture == 'multi' and single_score > 0.60:
        confidence -= 0.10

    complexity_bias = {
        'trivial':  -0.05,
        'simple':    0.00,
        'moderate':  0.05,
        'complex':   0.10,
    }[result.task_complexity]

    confidence += complexity_bias + adjustment

    if result.task_type == 'atomic':
        confidence += 0.05
    elif result.task_type == 'ambiguous':
        confidence = min(confidence, 0.65 + 0.1 * (confidence - 0.65))

    return round(max(0.0, min(1.0, confidence)), 4)
