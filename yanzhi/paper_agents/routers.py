from .parameters import GraphState


def citation_router(state: GraphState) -> str:
    if state['paper']['add_citations'] is True:
        return 'citations_node'
    elif state['paper']['add_citations'] is False:
        return '__end__'
    else:
        raise Exception('add_citations 参数值错误')
