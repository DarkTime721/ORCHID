import json
import statistics
from typing import Optional

from models.state import RoutingHistorySummary


def load_routing_history(task_type: str) -> Optional[RoutingHistorySummary]:
    try:
        with open("run_history.json", "r") as f:
            all_runs = [json.loads(line) for line in f]
    except FileNotFoundError:
        return None

    relevant = [
        r for r in all_runs if r.get('task_type') == task_type
    ]

    if len(relevant) < 5:
        return None

    single_runs = [r for r in relevant if 'single' in r['architectures_run']]
    multi_runs = [r for r in relevant if 'multi' in r['architectures_run']]

    if len(single_runs) < 2 or len(multi_runs) < 2:
        return None

    single_success_rate = (
        statistics.mean([r['success'] for r in single_runs if r['success'] is not None])
        if single_runs else 0.0
    )

    multi_success_rate = (
        statistics.mean([r['success'] for r in multi_runs if r['success'] is not None])
        if multi_runs else 0.0
    )

    delta = multi_success_rate - single_success_rate
    adjustment = max(-0.2, min(0.2, delta * 0.3))

    return RoutingHistorySummary(
        task_type=task_type,
        total_runs=len(relevant),
        single_success_rate=single_success_rate,
        multi_success_rate=multi_success_rate,
        recommended_adjustment=adjustment
    )
