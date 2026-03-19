"""实验进度日志：每个节点写入 progress.jsonl，UI 实时读取展示。"""
import json
import time
import os
from typing import Literal

# 完整的节点执行顺序（用于 UI 绘制进度条）
STEP_ORDER = [
    ("preprocess",       "初始化"),
    ("dataset_suggest",  "推荐数据集"),
    ("data_fetch",       "下载数据集"),
    ("criteria_infer",   "推断验收标准"),
    ("criteria_confirm", "等待用户确认标准"),
    ("code_gen",         "生成实验代码"),
    ("code_execute",     "执行代码"),
    ("fix_code",         "修复代码"),
    ("diagnose",         "诊断问题"),
    ("lit_search",       "文献检索"),
    ("lit_review",       "等待用户决策"),
    ("method_refine",    "优化研究方法"),
    ("save_results",     "保存结果"),
]

STEP_LABELS = {k: v for k, v in STEP_ORDER}

Status = Literal["start", "done", "error", "interrupt"]


def write_progress(module_folder: str, node: str,
                   status: Status, msg: str = "") -> None:
    """追加一条进度记录到 progress.jsonl。"""
    record = {
        "ts":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "node": node,
        "status": status,
        "msg":  msg,
        "label": STEP_LABELS.get(node, node),
    }
    path = os.path.join(module_folder, "progress.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_progress(module_folder: str) -> list[dict]:
    """读取所有进度记录，不存在则返回空列表。"""
    path = os.path.join(module_folder, "progress.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def clear_progress(module_folder: str) -> None:
    """清除进度文件（新一轮实验前调用）。"""
    path = os.path.join(module_folder, "progress.jsonl")
    if os.path.exists(path):
        os.remove(path)


def summarize_progress(records: list[dict]) -> dict:
    """
    返回当前进度摘要：
    {
      "completed": ["preprocess", "criteria_infer", ...],
      "current":   "code_gen",      # 最后一个 start 但未 done 的节点
      "current_label": "生成实验代码",
      "errors":    [...],
      "outer_iter": 1,              # 外层轮次（code_gen 出现次数）
      "inner_iter": 0,              # 内层轮次（fix_code 出现次数）
    }
    """
    completed = []
    node_status: dict[str, str] = {}
    outer_iter = 0
    inner_iter = 0

    for r in records:
        node   = r["node"]
        status = r["status"]
        node_status[node] = status
        if status == "done":
            if node not in completed:
                completed.append(node)
            if node == "code_gen":
                outer_iter += 1
            if node == "fix_code":
                inner_iter += 1

    current = None
    for r in reversed(records):
        if r["status"] == "start":
            if node_status.get(r["node"]) == "start":
                current = r["node"]
                break

    errors = [r for r in records if r["status"] == "error"]

    return {
        "completed":     completed,
        "current":       current,
        "current_label": STEP_LABELS.get(current, current) if current else None,
        "errors":        errors,
        "outer_iter":    outer_iter,
        "inner_iter":    inner_iter,
    }
