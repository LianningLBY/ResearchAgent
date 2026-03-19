"""LangGraph 实验工作流（双层循环 + Human-in-the-Loop）。"""
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .parameters import ExperimentState
from .experiment_node import (
    preprocess_node,
    dataset_router,
    dataset_suggest_node,
    data_fetch_node,
    criteria_infer_node,
    code_gen_node,
    fix_code_node,
    code_execute_node,
    inner_router,
    diagnose_node,
    outer_router,
    lit_search_node,
    lit_review_node,
    human_decision_router,
    method_refine_node,
    save_results_node,
)


def build_experiment_graph():
    """构建 HITL 自动实验工作流。

    流程：
        START → preprocess → dataset_router
            → [有本地数据] → criteria_infer
            → [无本地数据] → dataset_suggest [interrupt③: 用户选数据集]
                           → data_fetch（下载 + schema 描述）
                           → criteria_infer [interrupt①: 用户确认标准]
              → code_gen → code_execute → inner_router
                  [fix_code] → fix_code → code_execute（内层循环，最多 max_inner_iter 次）
                  [diagnose] → diagnose_node → outer_router
                      [save]       → save_results → END
                      [lit_search] → lit_search → lit_review [interrupt②: 用户决策]
                          [stop]            → save_results → END
                          [continue/modify] → method_refine → code_gen（外层循环）
    """
    builder = StateGraph(ExperimentState)

    builder.add_node("preprocess",      preprocess_node)
    builder.add_node("dataset_suggest", dataset_suggest_node)
    builder.add_node("data_fetch",      data_fetch_node)
    builder.add_node("criteria_infer",  criteria_infer_node)
    builder.add_node("code_gen",        code_gen_node)
    builder.add_node("fix_code",        fix_code_node)
    builder.add_node("code_execute",    code_execute_node)
    builder.add_node("diagnose",        diagnose_node)
    builder.add_node("lit_search",      lit_search_node)
    builder.add_node("lit_review",      lit_review_node)
    builder.add_node("method_refine",   method_refine_node)
    builder.add_node("save_results",    save_results_node)

    builder.add_edge(START, "preprocess")
    builder.add_conditional_edges(
        "preprocess",
        dataset_router,
        {"dataset_suggest": "dataset_suggest", "criteria_infer": "criteria_infer"},
    )
    builder.add_edge("dataset_suggest", "data_fetch")
    builder.add_edge("data_fetch",      "criteria_infer")
    builder.add_edge("criteria_infer",  "code_gen")
    builder.add_edge("code_gen",        "code_execute")
    builder.add_conditional_edges(
        "code_execute",
        inner_router,
        {"fix_code": "fix_code", "diagnose": "diagnose"},
    )
    builder.add_edge("fix_code",        "code_execute")
    builder.add_conditional_edges(
        "diagnose",
        outer_router,
        {"save_results": "save_results", "lit_search": "lit_search"},
    )
    builder.add_edge("lit_search",      "lit_review")
    builder.add_conditional_edges(
        "lit_review",
        human_decision_router,
        {"save_results": "save_results", "method_refine": "method_refine"},
    )
    builder.add_edge("method_refine",   "code_gen")
    builder.add_edge("save_results",    END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
