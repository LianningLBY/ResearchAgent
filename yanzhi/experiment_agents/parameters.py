from typing_extensions import TypedDict, Any
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from ..key_manager import KeyManager


class FILES_EXP(TypedDict):
    Folder: str
    data_description: str
    methods: str
    code_file: str
    exec_log: str
    plots_dir: str
    results_md: str
    LLM_calls: str
    Error: str
    f_stream: str
    module_folder: str


class TOKENS(TypedDict):
    ti: int
    to: int
    i:  int
    o:  int


class LLM_EXP(TypedDict):
    model: str
    max_output_tokens: int
    llm: Any
    temperature: float
    stream_verbose: bool


class ExperimentState(TypedDict):
    messages:          Annotated[list[AnyMessage], add_messages]
    llm:               LLM_EXP
    tokens:            TOKENS
    files:             FILES_EXP
    keys:              KeyManager
    data_description:  str
    methods:           str
    # ── 验收标准 ──────────────────────────────────────
    criteria:          str   # LLM 推断的验收标准（自然语言）
    # ── 代码与执行 ────────────────────────────────────
    code:              str
    exec_output:       str
    exec_success:      bool
    # ── 迭代计数 ──────────────────────────────────────
    inner_iteration:   int   # 内层代码修复轮次
    outer_iteration:   int   # 外层方法优化轮次
    max_inner_iter:    int   # 最大内层轮次（默认 3）
    max_outer_iter:    int   # 最大外层轮次（默认 2）
    timeout:           int
    # ── 诊断与文献 ────────────────────────────────────
    failure_type:      str   # "code_error" | "insufficient"
    diagnosis:         str   # 具体诊断描述
    search_query:      str   # 文献检索词（英文）
    literature_found:  str   # 找到的文献摘要
    # ── 人工介入 ──────────────────────────────────────
    human_decision:    str   # "continue" | "stop" | "modify"
    human_input:       str   # 用户的自定义方向（modify 时）
    # ── 数据集发现 ────────────────────────────────────
    dataset_candidates:   list   # list[DatasetCandidate.to_dict()]
    chosen_dataset:       dict   # DatasetCandidate.to_dict()，用户选中的
    dataset_local_path:   str    # 下载后的本地路径（None → 合成数据）
    skip_dataset_search:  bool   # True = 用户已提供本地数据，跳过搜索
    # ── 最终结果 ──────────────────────────────────────
    result_summary:    str
