from models.state import TaskState

CONFIDENCE_THRESHOLD = 0.75


def confidence_router(state: TaskState) -> str:
    if state['confidence'] is None:
        return ['single', 'multi']

    if state['confidence'] >= CONFIDENCE_THRESHOLD:
        return state['recommended_architecture']

    return ['single', 'multi']
