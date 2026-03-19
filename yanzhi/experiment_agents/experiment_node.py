"""LangGraph 节点：双层循环 + Human-in-the-Loop 实验流程。"""
import os
import re
import json
import shutil
import time
from pathlib import Path

import json5
import requests
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ..paper_agents.tools import LLM_call
from ..llm_factory import build_llm
from ..log import get_logger
from ..config import SS_API_MAX_RETRIES, SS_API_RETRY_DELAY
from .parameters import ExperimentState
from .executor import execute_code
from .progress import write_progress, clear_progress
from .dataset_finder import search_datasets, DatasetCandidate
from .data_fetch import fetch_dataset
from .prompts import (
    data_requirements_prompt,
    criteria_infer_prompt, code_gen_prompt, fix_code_prompt,
    diagnose_prompt, method_refine_prompt, results_summary_prompt,
)

logger = get_logger(__name__)


def _prog(state: ExperimentState, node: str, status: str, msg: str = "") -> None:
    """写进度记录（module_folder 可能还未设置时静默忽略）。"""
    folder = state.get("files", {}).get("module_folder", "")
    if folder:
        write_progress(folder, node, status, msg)


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _extract_code(text: str) -> str | None:
    m = re.search(r"\\begin\{CODE\}(.*?)\\end\{CODE\}", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return None


def _parse_json(text: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text.strip()
    return json5.loads(raw)


def _ss_search(query: str, limit: int = 8) -> str:
    """调用 Semantic Scholar，返回格式化的文献摘要字符串。"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit,
              "fields": "title,authors,year,abstract,url"}
    delay = SS_API_RETRY_DELAY
    for attempt in range(SS_API_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                papers = resp.json().get("data", [])
                if not papers:
                    return "未找到相关文献。"
                lines = []
                for i, p in enumerate(papers, 1):
                    authors = ", ".join(a.get("name", "") for a in p.get("authors", []))
                    lines.append(
                        f"{i}. {p.get('title','无标题')} ({p.get('year','?')})\n"
                        f"   作者：{authors}\n"
                        f"   摘要：{p.get('abstract','无摘要')[:300]}\n"
                        f"   URL：{p.get('url','')}\n"
                    )
                return "\n".join(lines)
            if resp.status_code == 429:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            return f"API 请求失败（状态码 {resp.status_code}）"
        except requests.RequestException as e:
            logger.warning("Semantic Scholar 网络异常（%d/%d）：%s", attempt+1, SS_API_MAX_RETRIES, e)
            time.sleep(delay); delay = min(delay * 2, 60)
    return "文献检索失败（达到最大重试次数）"


# ─── 预处理节点 ──────────────────────────────────────────────────────────────

def preprocess_node(state: ExperimentState, config: RunnableConfig):
    state["tokens"] = {"ti": 0, "to": 0, "i": 0, "o": 0}
    state["llm"]["llm"] = build_llm(
        model=state["llm"]["model"],
        temperature=state["llm"]["temperature"],
        key_manager=state["keys"],
    )

    folder = state["files"]["Folder"]
    module_folder = os.path.join(folder, "experiment_output")
    os.makedirs(module_folder, exist_ok=True)

    plots_dir = state["files"].get("plots_dir") or os.path.join(folder, "input_files", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    state["files"] = {
        **state["files"],
        "module_folder": module_folder,
        "code_file":     os.path.join(module_folder, "experiment.py"),
        "exec_log":      os.path.join(module_folder, "exec.log"),
        "results_md":    os.path.join(folder, "input_files", "results.md"),
        "LLM_calls":     os.path.join(module_folder, "LLM_calls.txt"),
        "Error":         os.path.join(module_folder, "Error.txt"),
        "f_stream":      os.path.join(module_folder, "stream.log"),
        "plots_dir":     plots_dir,
    }

    for key in ["LLM_calls", "Error", "exec_log"]:
        fp = state["files"][key]
        if os.path.exists(fp):
            os.remove(fp)
    clear_progress(module_folder)  # 清除上轮进度
    write_progress(module_folder, "preprocess", "start", "初始化实验环境")

    desc_path = state["files"]["data_description"]
    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"未找到数据描述文件：{desc_path}")
    with open(desc_path, "r", encoding="utf-8") as f:
        data_description = f.read()

    methods_path = state["files"]["methods"]
    if not os.path.exists(methods_path):
        raise FileNotFoundError(f"未找到研究方法文件：{methods_path}")
    with open(methods_path, "r", encoding="utf-8") as f:
        methods = f.read()

    result_state = {
        **state,
        "data_description": data_description,
        "methods":          methods,
        "code":             "",
        "exec_output":      "",
        "exec_success":     False,
        "result_summary":   "",
        "criteria":         "",
        "inner_iteration":  0,
        "outer_iteration":  0,
        "failure_type":       "",
        "diagnosis":          "",
        "search_query":       "",
        "literature_found":   "",
        "human_decision":     "",
        "human_input":        "",
        "dataset_candidates": [],
        "chosen_dataset":     {},
        "dataset_local_path": "",
        "skip_dataset_search": False,
    }
    write_progress(module_folder, "preprocess", "done", "初始化完成")
    return result_state


# ─── 数据集路由（判断是否需要搜索）──────────────────────────────────────────────

def _has_local_data(data_description: str) -> bool:
    """检查 data_description 中是否已经有可用的本地文件路径。"""
    import re as _re
    # 匹配 /绝对路径/文件.扩展名
    pattern = r"/[^\s\"']+\.(?:csv|parquet|json|jsonl|tsv|xlsx|npy|npz|h5|hdf5)\b"
    for match in _re.finditer(pattern, data_description, _re.IGNORECASE):
        if os.path.exists(match.group()):
            return True
    return False


def dataset_router(state: ExperimentState) -> str:
    """preprocess 后：有本地数据 → 跳过搜索；无 → 搜索推荐。"""
    if state.get("skip_dataset_search"):
        return "criteria_infer"
    if _has_local_data(state.get("data_description", "")):
        logger.info("检测到本地数据文件，跳过数据集搜索。")
        return "criteria_infer"
    return "dataset_suggest"


# ─── 数据集推荐节点（含介入点③）────────────────────────────────────────────────

def dataset_suggest_node(state: ExperimentState, config: RunnableConfig):
    """提取数据需求 → 搜索候选数据集 → 等待用户选择。"""
    _prog(state, "dataset_suggest", "start", "分析数据需求")
    logger.info("分析研究方法中的数据需求...")

    # 用 LLM 提取数据需求
    prompt = data_requirements_prompt(state["methods"])
    state, result = LLM_call(prompt, state)

    try:
        parsed       = _parse_json(result)
        search_query = parsed.get("search_query", "")
        domain       = parsed.get("domain", "")
        needs_local  = parsed.get("needs_local_data", False)
        reason       = parsed.get("reason", "")
    except Exception:
        search_query = state["methods"][:60]
        domain       = ""
        needs_local  = False
        reason       = ""

    _prog(state, "dataset_suggest", "start", f"搜索数据集：{search_query}")

    if needs_local:
        # LLM 认为可能需要私有数据，但仍先搜索公开数据集供用户选择
        logger.info("LLM 提示可能需要私有数据，仍自动搜索公开数据集：%s", search_query)

    # 搜索公开数据集（网络错误时降级为只含合成数据）
    try:
        candidates = search_datasets(search_query)
    except Exception as e:
        logger.warning("数据集搜索网络异常，降级为合成数据：%s", e)
        from .dataset_finder import _SYNTHETIC
        candidates = [_SYNTHETIC]
    candidates_dicts = [c.to_dict() for c in candidates]

    _prog(state, "dataset_suggest", "interrupt",
          f"找到 {len(candidates)} 个候选，等待用户选择")

    # ── 介入点③ ───────────────────────────────────────────────────────────────
    chosen = interrupt({
        "type":       "dataset_select",
        "domain":     domain,
        "reason":     reason,
        "search_query": search_query,
        "candidates": candidates_dicts,
        "message":    f"请从以下候选数据集中选择一个用于实验（领域：{domain}）。",
    })

    # chosen 可以是 dict（候选之一的 to_dict()）或 index（int）或 "synthetic"
    if isinstance(chosen, int) and 0 <= chosen < len(candidates_dicts):
        chosen_dict = candidates_dicts[chosen]
    elif isinstance(chosen, str) and chosen == "synthetic":
        chosen_dict = DatasetCandidate(
            "synthetic", "synthetic", "合成数据", "synthetic", "即时生成", []
        ).to_dict()
    elif isinstance(chosen, dict):
        chosen_dict = chosen
    else:
        # 默认选第一个非合成数据集，或合成数据
        chosen_dict = candidates_dicts[0] if candidates_dicts else {}

    _prog(state, "dataset_suggest", "done",
          f"用户选择：{chosen_dict.get('name', '?')}（{chosen_dict.get('source', '?')}）")
    logger.info("用户选择数据集：%s", chosen_dict.get("name"))
    return {**state, "dataset_candidates": candidates_dicts, "chosen_dataset": chosen_dict}


# ─── 数据集下载节点 ───────────────────────────────────────────────────────────

def data_fetch_node(state: ExperimentState, config: RunnableConfig):
    """下载选中的数据集，将 schema 追加到 data_description。"""
    chosen = state.get("chosen_dataset", {})
    if not chosen:
        return state

    _prog(state, "data_fetch", "start",
          f"下载 {chosen.get('name', '?')}（{chosen.get('source', '?')}）")
    logger.info("下载数据集：%s", chosen.get("name"))

    cache_dir  = os.path.join(state["files"]["Folder"], "datasets")
    candidate  = DatasetCandidate.from_dict(chosen)
    hf_token   = getattr(state.get("keys"), "HF_TOKEN", None) or None
    fetch_result = fetch_dataset(candidate, cache_dir, hf_token=hf_token)

    if fetch_result["success"]:
        local_path = fetch_result.get("local_path") or ""
        schema_md  = fetch_result.get("schema_md", "")
        enriched = state["data_description"]
        if schema_md:
            enriched += f"\n\n## 数据集信息（自动获取）\n{schema_md}"
        _prog(state, "data_fetch", "done",
              f"已下载：{os.path.basename(local_path) if local_path else '合成数据'}")
        return {**state, "data_description": enriched, "dataset_local_path": local_path}

    err = fetch_result.get("error", "未知错误")

    # ── gated 数据集但没有 token → HITL 让用户提供 token 或换数据集 ──────────
    if err == "gated_no_token":
        _prog(state, "data_fetch", "interrupt", "需要 HF Token，等待用户操作")
        decision = interrupt({
            "type":        "dataset_gated_token_required",
            "dataset":     chosen.get("name"),
            "access_note": fetch_result.get("access_note", ""),
            "message":     (
                f"数据集「{chosen.get('name')}」需要 HuggingFace Token 才能下载。\n"
                f"{fetch_result.get('access_note', '')}\n"
                "请选择：\n"
                "  1. 在系统设置中填入 HF Token 后重试（输入 'retry'）\n"
                "  2. 改用合成数据（输入 'synthetic'）\n"
                "  3. 选择其他数据集（输入候选数据集的序号，从 0 开始）"
            ),
            "candidates":  state.get("dataset_candidates", []),
        })
        if str(decision).strip().lower() == "retry":
            # 用户已在 settings 填入 token，重新读取并重试
            new_token = getattr(state.get("keys"), "HF_TOKEN", None) or None
            fetch_result2 = fetch_dataset(candidate, cache_dir, hf_token=new_token)
            if fetch_result2["success"]:
                local_path = fetch_result2.get("local_path") or ""
                schema_md  = fetch_result2.get("schema_md", "")
                enriched   = state["data_description"]
                if schema_md:
                    enriched += f"\n\n## 数据集信息（自动获取）\n{schema_md}"
                _prog(state, "data_fetch", "done", f"已下载（token 重试）：{chosen.get('name')}")
                return {**state, "data_description": enriched, "dataset_local_path": local_path}
        elif str(decision).strip().isdigit():
            idx = int(decision.strip())
            candidates = state.get("dataset_candidates", [])
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
                candidate2 = DatasetCandidate.from_dict(chosen)
                fetch_result2 = fetch_dataset(candidate2, cache_dir, hf_token=hf_token)
                if fetch_result2["success"]:
                    local_path = fetch_result2.get("local_path") or ""
                    schema_md  = fetch_result2.get("schema_md", "")
                    enriched   = state["data_description"]
                    if schema_md:
                        enriched += f"\n\n## 数据集信息（自动获取）\n{schema_md}"
                    _prog(state, "data_fetch", "done", f"已下载（备选）：{chosen.get('name')}")
                    return {**state, "data_description": enriched,
                            "dataset_local_path": local_path, "chosen_dataset": chosen}
        # 兜底：合成数据
        _prog(state, "data_fetch", "done", "使用合成数据")
        fallback = state["data_description"] + "\n\n（数据集受限，请在代码中生成合成数据）"
        return {**state, "data_description": fallback, "dataset_local_path": ""}

    # ── 私有数据集 → HITL：修改研究想法 或 手动上传 ─────────────────────────
    if err == "private_dataset" or err == "unknown":
        _prog(state, "data_fetch", "interrupt", "数据集不可用，等待用户决策")
        decision = interrupt({
            "type":    "dataset_unavailable",
            "dataset": chosen.get("name"),
            "message": (
                f"数据集「{chosen.get('name')}」为私有数据集，无法自动下载。\n"
                "请选择：\n"
                "  A. 修改研究想法，转向有公开数据集支持的方向（输入 'modify_idea'）\n"
                "  B. 手动上传数据集文件（在系统设置→上传数据文件后输入 'uploaded'）\n"
                "  C. 改用合成数据继续实验（输入 'synthetic'）"
            ),
        })
        decision_str = str(decision).strip().lower()
        if decision_str == "modify_idea":
            _prog(state, "data_fetch", "done", "用户选择修改研究想法")
            return {**state, "human_decision": "modify_idea", "dataset_local_path": ""}
        elif decision_str == "uploaded":
            # 检查用户是否上传了文件
            data_dir = os.path.join(state["files"]["Folder"], "data")
            uploaded_files = []
            if os.path.exists(data_dir):
                uploaded_files = [f for f in os.listdir(data_dir)
                                  if f.endswith((".csv", ".parquet", ".json", ".tsv", ".xlsx"))]
            if uploaded_files:
                local_path = os.path.join(data_dir, uploaded_files[0])
                from .data_fetch import _describe_file
                schema_md = _describe_file(local_path)
                enriched  = state["data_description"]
                if schema_md:
                    enriched += f"\n\n## 数据集信息（用户上传）\n{schema_md}"
                _prog(state, "data_fetch", "done", f"使用上传文件：{uploaded_files[0]}")
                return {**state, "data_description": enriched, "dataset_local_path": local_path}

    # 最终兜底：合成数据
    _prog(state, "data_fetch", "error", f"下载失败（{err}），使用合成数据")
    logger.warning("数据集下载失败（%s），回退到合成数据。", err)
    fallback = state["data_description"] + "\n\n（数据集下载失败，请在代码中生成合成数据）"
    return {**state, "data_description": fallback, "dataset_local_path": ""}


# ─── 验收标准推断节点（含介入点①）─────────────────────────────────────────────

def criteria_infer_node(state: ExperimentState, config: RunnableConfig):
    _prog(state, "criteria_infer", "start", "推断验收标准")
    logger.info("推断验收标准...")
    prompt = criteria_infer_prompt(state["methods"], state["data_description"])
    state, criteria_text = LLM_call(prompt, state)
    _prog(state, "criteria_infer", "done", criteria_text[:80])
    logger.info("推断完成，等待用户确认...")

    _prog(state, "criteria_confirm", "interrupt", "等待用户确认")
    confirmed = interrupt({
        "type":     "criteria_confirm",
        "criteria": criteria_text,
        "message":  "系统根据研究方法推断了以下验收标准，请确认或修改后点击继续。",
    })
    final_criteria = confirmed if isinstance(confirmed, str) and confirmed.strip() else criteria_text
    _prog(state, "criteria_confirm", "done", "用户已确认")
    logger.info("用户确认验收标准。")
    return {**state, "criteria": final_criteria}


# ─── 代码生成节点 ─────────────────────────────────────────────────────────────

def code_gen_node(state: ExperimentState, config: RunnableConfig):
    outer = state.get("outer_iteration", 0) + 1
    _prog(state, "code_gen", "start", f"生成实验代码（第{outer}轮）")
    logger.info("代码生成（外层第 %d 轮）", outer)
    # 只在内层重试时传入历史输出，外层新一轮从头生成
    prev_out  = state.get("exec_output", "") if state.get("inner_iteration", 0) > 0 else ""
    prev_code = state.get("code", "")        if state.get("inner_iteration", 0) > 0 else ""
    prompt = code_gen_prompt(
        data_description=state["data_description"],
        methods=state["methods"],
        criteria=state.get("criteria", ""),
        prev_output=prev_out,
        prev_code=prev_code,
    )
    state, result = LLM_call(prompt, state)
    code = _extract_code(result) or result.strip()
    with open(state["files"]["code_file"], "w", encoding="utf-8") as f:
        f.write(code)
    _prog(state, "code_gen", "done", f"{len(code)} 字符代码")
    return {**state, "code": code, "inner_iteration": 0}


# ─── 代码修复节点（内层）─────────────────────────────────────────────────────

def fix_code_node(state: ExperimentState, config: RunnableConfig):
    inner = state.get("inner_iteration", 0)
    _prog(state, "fix_code", "start", f"修复代码（第{inner+1}次）")
    logger.info("代码修复（内层第 %d 次）", inner + 1)
    prompt = fix_code_prompt(state["code"], state["exec_output"])
    state, result = LLM_call(prompt, state)
    code = _extract_code(result) or state["code"]
    with open(state["files"]["code_file"], "w", encoding="utf-8") as f:
        f.write(code)
    _prog(state, "fix_code", "done", f"{len(code)} 字符")
    return {**state, "code": code, "inner_iteration": inner + 1}


# ─── 代码执行节点 ─────────────────────────────────────────────────────────────

def code_execute_node(state: ExperimentState, config: RunnableConfig):
    inner = state.get("inner_iteration", 0)
    _prog(state, "code_execute", "start", f"执行代码（内层第{inner+1}次）")
    logger.info("代码执行（内层第 %d 次）", inner + 1)
    result = execute_code(
        code=state["code"],
        work_dir=state["files"]["module_folder"],
        timeout=state.get("timeout", 120),
        plots_dir=state["files"]["plots_dir"],
    )
    output = result.combined_output
    if result.plot_files:
        output += f"\n[已生成 {len(result.plot_files)} 张图片：{', '.join(Path(p).name for p in result.plot_files)}]"

    with open(state["files"]["exec_log"], "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n外层{state.get('outer_iteration',0)+1} 内层{inner+1}\n{'='*60}\n{output}\n")

    status_msg = f"成功，{len(result.plot_files)}张图片" if result.success else f"失败（returncode={result.returncode}）"
    _prog(state, "code_execute", "done" if result.success else "error", status_msg)
    logger.info("执行完成（returncode=%d）", result.returncode)
    return {**state, "exec_output": output, "exec_success": result.success}


# ─── 内层路由 ─────────────────────────────────────────────────────────────────

def inner_router(state: ExperimentState) -> str:
    if state.get("exec_success"):
        return "diagnose"
    if state.get("inner_iteration", 0) < state.get("max_inner_iter", 3):
        return "fix_code"
    logger.warning("代码修复达到内层上限（%d 次），转入诊断。", state.get("max_inner_iter", 3))
    return "diagnose"


# ─── 诊断节点 ─────────────────────────────────────────────────────────────────

def diagnose_node(state: ExperimentState, config: RunnableConfig):
    _prog(state, "diagnose", "start", "诊断实验问题")
    logger.info("诊断实验问题...")
    prompt = diagnose_prompt(
        exec_output=state["exec_output"],
        methods=state["methods"],
        criteria=state.get("criteria", ""),
        exec_success=state.get("exec_success", False),
    )
    state, result = LLM_call(prompt, state)
    try:
        parsed       = _parse_json(result)
        failure_type = parsed.get("failure_type", "insufficient")
        diagnosis    = parsed.get("diagnosis", "")
        search_query = parsed.get("search_query", "")
    except Exception as e:
        logger.warning("诊断 JSON 解析失败（%s）。", e)
        failure_type = "insufficient"
        diagnosis    = "结果不足"
        search_query = "research methodology improvement"

    # 执行成功且代码能跑 → 检查是否真的不满足
    # 如果 exec_success=True 且 failure_type 被解析为 "satisfied"，直接保存
    if state.get("exec_success") and failure_type == "satisfied":
        return {**state, "failure_type": "satisfied",
                "diagnosis": diagnosis, "search_query": search_query}

    _prog(state, "diagnose", "done", f"{failure_type}: {diagnosis[:60]}")
    logger.info("诊断：%s | %s | 检索词：%s", failure_type, diagnosis, search_query)
    return {**state, "failure_type": failure_type,
            "diagnosis": diagnosis, "search_query": search_query}


# ─── 外层路由（诊断后）────────────────────────────────────────────────────────

def outer_router(state: ExperimentState) -> str:
    if state.get("failure_type") == "satisfied":
        return "save_results"
    outer = state.get("outer_iteration", 0)
    max_o = state.get("max_outer_iter", 2)
    if outer >= max_o:
        logger.info("外层优化达到上限（%d 轮），保存当前最优结果。", max_o)
        return "save_results"
    # code_error：代码本身有 bug，直接重新生成代码，不需要查文献
    if state.get("failure_type") == "code_error":
        logger.info("代码错误，跳过文献检索，直接重新生成代码。")
        return "code_gen"
    # insufficient：结果质量不足，查文献改进研究方法
    return "lit_search"


# ─── 文献搜索节点 ─────────────────────────────────────────────────────────────

def lit_search_node(state: ExperimentState, config: RunnableConfig):
    query = state.get("search_query", "research methodology improvement")
    _prog(state, "lit_search", "start", f"检索：{query[:60]}")
    logger.info("文献检索：%s", query)
    literature = _ss_search(query, limit=8)
    _prog(state, "lit_search", "done", f"找到文献（{len(literature)} 字符）")
    logger.info("检索完成（%d 字符）", len(literature))
    return {**state, "literature_found": literature}


# ─── 文献评审节点（介入点②）─────────────────────────────────────────────────

def lit_review_node(state: ExperimentState, config: RunnableConfig):
    outer = state.get("outer_iteration", 0)
    _prog(state, "lit_review", "interrupt", f"等待用户决策（第{outer+1}轮）")
    logger.info("等待用户决策（外层第 %d 轮）...", outer + 1)

    # ── 介入点② ──────────────────────────────────────────────────────────────
    user_response = interrupt({
        "type":         "lit_review",
        "outer":        outer + 1,
        "diagnosis":    state.get("diagnosis", ""),
        "search_query": state.get("search_query", ""),
        "literature":   state.get("literature_found", ""),
        "message":      f"第 {outer+1} 轮结果不足，已找到相关文献，请决定下一步。",
    })

    if isinstance(user_response, dict):
        decision    = user_response.get("decision", "continue")
        human_input = user_response.get("input", "")
    else:
        decision    = str(user_response)
        human_input = ""

    _prog(state, "lit_review", "done", f"用户决策：{decision}")
    logger.info("用户决策：%s", decision)
    return {
        **state,
        "human_decision":  decision,
        "human_input":     human_input,
        "outer_iteration": outer + 1,
    }


# ─── 人工决策路由 ─────────────────────────────────────────────────────────────

def human_decision_router(state: ExperimentState) -> str:
    if state.get("human_decision") == "stop":
        return "save_results"
    return "method_refine"


# ─── 方法优化节点 ─────────────────────────────────────────────────────────────

def method_refine_node(state: ExperimentState, config: RunnableConfig):
    outer = state.get("outer_iteration", 0)
    _prog(state, "method_refine", "start", f"优化研究方法（第{outer}轮）")
    logger.info("方法优化（外层第 %d 轮）", outer)
    prompt = method_refine_prompt(
        methods=state["methods"],
        diagnosis=state.get("diagnosis", ""),
        literature_found=state.get("literature_found", ""),
        human_input=state.get("human_input", ""),
    )
    state, result = LLM_call(prompt, state)
    m = re.search(r"\\begin\{METHODS\}(.*?)\\end\{METHODS\}", result, re.DOTALL)
    new_methods = m.group(1).strip() if m else state["methods"]
    with open(state["files"]["methods"], "w", encoding="utf-8") as f:
        f.write(new_methods)
    _prog(state, "method_refine", "done", f"方法已更新（{len(new_methods)} 字符）")
    logger.info("研究方法已更新（%d 字符）", len(new_methods))
    return {**state, "methods": new_methods}


# ─── 结果保存节点 ─────────────────────────────────────────────────────────────

def save_results_node(state: ExperimentState, config: RunnableConfig):
    _prog(state, "save_results", "start", "保存实验结果")
    logger.info("保存实验结果...")
    if state.get("exec_output"):
        prompt = results_summary_prompt(
            exec_output=state["exec_output"],
            methods=state["methods"],
            data_description=state["data_description"],
        )
        state, summary_text = LLM_call(prompt, state)
    else:
        summary_text = "实验未产生有效输出。"

    results_path = state["files"]["results_md"]
    if os.path.exists(results_path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        history_dir = os.path.join(state["files"]["Folder"], "history")
        os.makedirs(history_dir, exist_ok=True)
        shutil.copy2(results_path, os.path.join(history_dir, f"results_{ts}.md"))

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    total = state["tokens"]["ti"] + state["tokens"]["to"]
    _prog(state, "save_results", "done", f"实验完成，共 {total} tokens")
    logger.info("实验完成。总 tokens：%d（输入 %d，输出 %d）",
                total, state["tokens"]["ti"], state["tokens"]["to"])
    return {**state, "result_summary": summary_text}
