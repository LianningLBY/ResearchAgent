from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .parameters import GraphState
from .paper_node import (abstract_node, citations_node, conclusions_node,
                         introduction_node, keywords_node, methods_node,
                         plots_node, refine_results, results_node)
from .reader import preprocess_node
from .routers import citation_router


def build_graph(mermaid_diagram=False):
    """构建中文论文生成工作流"""

    builder = StateGraph(GraphState)

    builder.add_node("preprocess_node",   preprocess_node)
    builder.add_node("keywords_node",     keywords_node)
    builder.add_node("abstract_node",     abstract_node)
    builder.add_node("introduction_node", introduction_node)
    builder.add_node("methods_node",      methods_node)
    builder.add_node("results_node",      results_node)
    builder.add_node("conclusions_node",  conclusions_node)
    builder.add_node("plots_node",        plots_node)
    builder.add_node("refine_results",    refine_results)
    builder.add_node("citations_node",    citations_node)

    builder.add_edge(START,                         "preprocess_node")
    builder.add_edge("preprocess_node",             "keywords_node")
    builder.add_edge("keywords_node",               "abstract_node")
    builder.add_edge("abstract_node",               "introduction_node")
    builder.add_edge("introduction_node",           "methods_node")
    builder.add_edge("methods_node",                "results_node")
    builder.add_edge("results_node",                "conclusions_node")
    builder.add_edge("conclusions_node",            "plots_node")
    builder.add_edge("plots_node",                  "refine_results")
    builder.add_conditional_edges("refine_results", citation_router)
    builder.add_edge("citations_node",              END)

    memory = MemorySaver()
    graph  = builder.compile(checkpointer=memory)

    if mermaid_diagram:
        try:
            import requests as req
            orig = req.post
            def patched(*args, **kwargs):
                kwargs.setdefault("timeout", 30)
                return orig(*args, **kwargs)
            req.post = patched
            img = graph.get_graph(xray=True).draw_mermaid_png()
            with open("graph_diagram.png", "wb") as f:
                f.write(img)
            print("图谱已保存至 graph_diagram.png")
        except Exception as e:
            print(f"生成图谱失败：{e}")

    return graph
