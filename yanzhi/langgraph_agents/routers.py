from .parameters import GraphState


def task_router(state: GraphState) -> str:
    if state['task'] == 'idea_generation':
        return 'maker'
    elif state['task'] == 'methods_generation':
        return 'methods'
    elif state['task'] == 'literature':
        return 'novelty'
    elif state['task'] == 'referee':
        return 'referee'
    else:
        raise Exception('任务类型错误！')


def router(state: GraphState) -> str:
    if state['idea']['iteration'] < state['idea']['total_iterations']:
        return "hater"
    else:
        return "__end__"


def literature_router(state: GraphState) -> str:
    return state['literature']['next_agent']
