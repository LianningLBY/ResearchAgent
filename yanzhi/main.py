import json
import os
import asyncio
import shutil
import threading
import time
from pathlib import Path

from .config import (DEFAUL_PROJECT_NAME, INPUT_FILES, PLOTS_FOLDER, HISTORY_DIR,
                     DESCRIPTION_FILE, IDEA_FILE, METHOD_FILE, RESULTS_FILE, LITERATURE_FILE,
                     HISTORY_PREVIEW_CHARS, DEFAULT_EXEC_TIMEOUT, DEFAULT_EXP_MAX_ITER)
from .research import Research
from .key_manager import KeyManager
from .llm import LLM, models
from .llm_factory import build_llm
from .log import get_logger
from .paper_agents.journal import Journal
from .paper_agents.agents_graph import build_graph
from .langgraph_agents.agents_graph import build_lg_graph


# ---- 工具函数（内联，避免触发 denario 重依赖） ----
import re
import warnings

logger = get_logger(__name__)


def _input_check(str_input: str) -> str:
    if str_input.endswith(".md"):
        with open(str_input, 'r', encoding='utf-8') as f:
            return f.read()
    return str_input


def _llm_parser(llm):
    if isinstance(llm, str):
        if llm not in models:
            raise KeyError(f"模型 '{llm}' 不可用。请从以下选择：{list(models.keys())}")
        return models[llm]
    return llm


def _check_file_paths(content: str) -> None:
    pattern = r'-\s*([^\n]+\.(?:csv|txt|md|py|json|yaml|yml|xml))'
    matches = re.findall(pattern, content, re.IGNORECASE)
    missing = [m.strip() for m in matches if not (os.path.exists(m.strip()) and os.path.isabs(m.strip()))]
    if missing:
        warnings.warn(f"以下文件路径不存在或非绝对路径：{missing}\n请使用格式：- /绝对/路径/文件.csv")


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except (ImportError, AttributeError):
        return False
    return True


class YanZhi:
    """
    中文科研助手主类。

    仿照 Denario 架构构建，全流程使用中文提示词，支持中文期刊 LaTeX 格式（ctex）。
    使用纯 LangGraph 后端，无需 cmbagent 依赖。

    Args:
        project_dir: 项目目录。若为 None，在当前目录下创建默认项目文件夹。
        clear_project_dir: 初始化时是否清空项目目录。
    """

    def __init__(self,
                 project_dir: str | None = None,
                 clear_project_dir: bool = False):

        if project_dir is None:
            project_dir = os.path.join(os.getcwd(), DEFAUL_PROJECT_NAME)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)

        self.research = Research()
        self.clear_project_dir = clear_project_dir

        if os.path.exists(project_dir) and clear_project_dir:
            shutil.rmtree(project_dir)
            os.makedirs(project_dir, exist_ok=True)

        self.project_dir  = project_dir
        self.plots_folder = os.path.join(self.project_dir, INPUT_FILES, PLOTS_FOLDER)
        os.makedirs(self.plots_folder, exist_ok=True)
        self._setup_input_files()

        self.keys = KeyManager()
        self.keys.get_keys_from_env()

        self.run_in_notebook = _in_notebook()
        self._exp_graphs: dict = {}   # thread_id → compiled experiment graph
        self.set_all()

    def _setup_input_files(self):
        input_files_dir = os.path.join(self.project_dir, INPUT_FILES)
        if os.path.exists(input_files_dir) and self.clear_project_dir:
            shutil.rmtree(input_files_dir)
        os.makedirs(input_files_dir, exist_ok=True)

    def _backup_file(self, filename: str) -> str | None:
        """将当前文件备份到 history/ 目录，文件名附加时间戳。"""
        src = os.path.join(self.project_dir, INPUT_FILES, filename)
        if not os.path.exists(src):
            return None
        history_dir = os.path.join(self.project_dir, HISTORY_DIR)
        os.makedirs(history_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = filename.replace(".md", "")
        dst = os.path.join(history_dir, f"{stem}_{timestamp}.md")
        shutil.copy2(src, dst)
        return dst

    def list_history(self, file_type: str) -> list:
        """
        列出某类文件的历史版本。

        Args:
            file_type: 文件名前缀，如 'idea'、'methods'、'literature'、'results'

        Returns:
            列表，每项为 dict {path, timestamp, preview, name}，按时间倒序
        """
        history_dir = os.path.join(self.project_dir, HISTORY_DIR)
        if not os.path.exists(history_dir):
            return []
        files = sorted(
            [f for f in os.listdir(history_dir)
             if f.startswith(file_type + "_") and f.endswith(".md")],
            reverse=True,
        )
        result = []
        for fname in files:
            fpath = os.path.join(history_dir, fname)
            ts = fname[len(file_type) + 1:].replace(".md", "")
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    preview = f.read()[:HISTORY_PREVIEW_CHARS]
            except Exception:
                preview = ""
            result.append({"path": fpath, "timestamp": ts, "preview": preview, "name": fname})
        return result

    # -----------------------------------------------------------------------
    # 输入设置
    # -----------------------------------------------------------------------

    def _setter(self, field, file_name):
        if field is None:
            try:
                with open(os.path.join(self.project_dir, INPUT_FILES, file_name), 'r', encoding='utf-8') as f:
                    field = f.read()
            except FileNotFoundError:
                raise FileNotFoundError("请提供内容字符串或 Markdown 文件路径。")
        field = _input_check(field)
        with open(os.path.join(self.project_dir, INPUT_FILES, file_name), 'w', encoding='utf-8') as f:
            f.write(field)
        return field

    def set_data_description(self, data_description: str | None = None) -> None:
        """设置数据和工具描述（字符串或 Markdown 文件路径）。"""
        self.research.data_description = self._setter(data_description, DESCRIPTION_FILE)
        _check_file_paths(self.research.data_description)

    def set_idea(self, idea: str | None = None) -> None:
        """手动设置研究想法。"""
        self.research.idea = self._setter(idea, IDEA_FILE)

    def set_method(self, method: str | None = None) -> None:
        """手动设置研究方法。"""
        self.research.methodology = self._setter(method, METHOD_FILE)

    def set_results(self, results: str | None = None) -> None:
        """手动设置研究结果。"""
        self.research.results = self._setter(results, RESULTS_FILE)

    def set_all(self) -> None:
        """自动读取项目目录中已有的各阶段文件。"""
        for setter in (self.set_data_description, self.set_idea,
                       self.set_method, self.set_results):
            try:
                setter()
            except FileNotFoundError:
                pass

    # -----------------------------------------------------------------------
    # 内容展示
    # -----------------------------------------------------------------------

    def _print(self, content: str) -> None:
        if self.run_in_notebook:
            from IPython.display import display, Markdown
            display(Markdown(content))
        else:
            print(content)

    def show_data_description(self) -> None:
        self._print(self.research.data_description)

    def show_idea(self) -> None:
        self._print(self.research.idea)

    def show_method(self) -> None:
        self._print(self.research.methodology)

    def show_results(self) -> None:
        self._print(self.research.results)

    # -----------------------------------------------------------------------
    # 生成阶段
    # -----------------------------------------------------------------------

    def get_idea(self,
                 llm: LLM | str = models["gemini-2.0-flash"],
                 iterations: int = 4,
                 verbose: bool = False) -> None:
        """
        生成研究想法（IdeaMaker/IdeaHater 对抗迭代）。

        Args:
            llm: 使用的 LLM 模型
            iterations: 迭代轮数（默认 4）
            verbose: 是否流式输出
        """
        self._backup_file(IDEA_FILE)
        logger.info("正在生成研究想法（%d 轮迭代）...", iterations)
        start_time = time.time()
        config     = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm        = _llm_parser(llm)
        graph      = build_lg_graph()

        f_desc = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        f_lit  = os.path.join(self.project_dir, INPUT_FILES, LITERATURE_FILE)
        input_state = {
            "task":  "idea_generation",
            "files": {"Folder": self.project_dir, "data_description": f_desc, "literature": f_lit},
            "llm":   {"model": llm.name, "temperature": llm.temperature,
                      "max_output_tokens": llm.max_output_tokens, "stream_verbose": verbose},
            "keys":  self.keys,
            "idea":  {"total_iterations": iterations},
        }
        graph.invoke(input_state, config)

        elapsed = time.time() - start_time
        logger.info("研究想法生成完成，用时 %d 分 %d 秒。", int(elapsed // 60), int(elapsed % 60))

    def check_idea(self,
                   llm: LLM | str = models["gemini-2.0-flash"],
                   max_iterations: int = 7,
                   verbose: bool = False) -> str:
        """
        通过 Semantic Scholar 查新，验证研究想法的新颖性。

        Args:
            llm: 使用的 LLM 模型
            max_iterations: 最大检索轮数
            verbose: 是否流式输出
        """
        self._backup_file(LITERATURE_FILE)
        logger.info("正在查新（最多 %d 轮）...", max_iterations)
        start_time = time.time()
        config     = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm        = _llm_parser(llm)
        graph      = build_lg_graph()

        f_desc = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        f_idea = os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE)
        input_state = {
            "task":       "literature",
            "files":      {"Folder": self.project_dir, "data_description": f_desc, "idea": f_idea},
            "llm":        {"model": llm.name, "temperature": llm.temperature,
                           "max_output_tokens": llm.max_output_tokens, "stream_verbose": verbose},
            "keys":       self.keys,
            "literature": {"max_iterations": max_iterations},
            "idea":       {"total_iterations": 4},
        }
        try:
            graph.invoke(input_state, config)
            elapsed = time.time() - start_time
            logger.info("查新完成，用时 %d 分 %d 秒。", int(elapsed // 60), int(elapsed % 60))
        except Exception as e:
            logger.error("查新失败：%s", e)
            return "查新过程中发生错误"

        try:
            with open(os.path.join(self.project_dir, INPUT_FILES, LITERATURE_FILE), 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "未找到文献查新结果文件"

    def suggest_literature_query(self,
                                 data_description: str | None = None,
                                 llm: LLM | str = models["gemini-2.0-flash"]) -> str:
        """
        基于研究方向生成用于文献检索的关键词/短语列表。

        Args:
            data_description: 研究方向文本；为空时从项目文件读取
            llm: 使用的 LLM 模型
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        if data_description is None:
            try:
                with open(os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE),
                          'r', encoding='utf-8') as f:
                    data_description = f.read()
            except FileNotFoundError:
                raise FileNotFoundError("未找到研究方向文件。")

        llm_obj = _llm_parser(llm)
        model = build_llm(model=llm_obj.name, temperature=0.3, key_manager=self.keys)

        messages = [
            SystemMessage(content=(
                "你是一位科研助理。请根据研究方向生成用于学术文献检索的关键词/短语列表。\n"
                "要求：3-6 个短语，优先英文，使用逗号分隔，只输出关键词列表。"
            )),
            HumanMessage(content=f"研究方向：\n{data_description}"),
        ]
        response = model.invoke(messages)
        return response.content.strip()

    def get_method(self,
                   llm: LLM | str = models["gemini-2.0-flash"],
                   verbose: bool = False) -> None:
        """
        生成研究方法论。

        Args:
            llm: 使用的 LLM 模型
            verbose: 是否流式输出
        """
        self._backup_file(METHOD_FILE)
        logger.info("正在生成研究方法...")
        start_time = time.time()
        config     = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm        = _llm_parser(llm)
        graph      = build_lg_graph()

        f_desc = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        f_idea = os.path.join(self.project_dir, INPUT_FILES, IDEA_FILE)
        input_state = {
            "task":  "methods_generation",
            "files": {"Folder": self.project_dir, "data_description": f_desc, "idea": f_idea},
            "llm":   {"model": llm.name, "temperature": llm.temperature,
                      "max_output_tokens": llm.max_output_tokens, "stream_verbose": verbose},
            "keys":  self.keys,
            "idea":  {"total_iterations": 4},
        }
        graph.invoke(input_state, config)

        elapsed = time.time() - start_time
        logger.info("研究方法生成完成，用时 %d 分 %d 秒。", int(elapsed // 60), int(elapsed % 60))

    def get_paper(self,
                  journal: Journal = Journal.NONE,
                  llm: LLM | str = models["gemini-2.0-flash"],
                  writer: str = "资深科研人员",
                  add_citations: bool = False) -> None:
        """
        生成中文学术论文（LaTeX 格式，支持 ctex 中文排版）。

        Args:
            journal: 期刊格式（NONE/CNKI/IEEE）
            llm: 使用的 LLM 模型
            writer: 作者身份设定（影响写作风格）
            add_citations: 是否添加参考文献（需要 Perplexity API Key）
        """
        logger.info("正在生成论文...")
        start_time = time.time()
        config     = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm        = _llm_parser(llm)
        graph      = build_graph()

        input_state = {
            "files": {"Folder": self.project_dir},
            "llm":   {"model": llm.name, "temperature": llm.temperature,
                      "max_output_tokens": llm.max_output_tokens},
            "paper": {"journal": journal, "add_citations": add_citations},
            "keys":  self.keys,
            "writer": writer,
        }
        asyncio.run(graph.ainvoke(input_state, config))

        elapsed = time.time() - start_time
        logger.info("论文生成完成，用时 %d 分 %d 秒。", int(elapsed // 60), int(elapsed % 60))

    def referee(self,
                llm: LLM | str = models["gemini-2.0-flash"],
                verbose: bool = False) -> None:
        """
        审阅已生成的论文，输出审稿报告。

        Args:
            llm: 使用的 LLM 模型
            verbose: 是否流式输出
        """
        logger.info("正在审阅论文...")
        start_time = time.time()
        config     = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
        llm        = _llm_parser(llm)
        graph      = build_lg_graph()

        f_desc = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        input_state = {
            "task":    "referee",
            "files":   {"Folder": self.project_dir, "data_description": f_desc},
            "llm":     {"model": llm.name, "temperature": llm.temperature,
                        "max_output_tokens": llm.max_output_tokens, "stream_verbose": verbose},
            "keys":    self.keys,
            "referee": {"paper_version": 2},
        }
        try:
            graph.invoke(input_state, config)
            elapsed = time.time() - start_time
            logger.info("论文审阅完成，用时 %d 分 %d 秒。", int(elapsed // 60), int(elapsed % 60))
        except FileNotFoundError as e:
            logger.error("未找到论文文件，请先运行 get_paper()。错误：%s", e)

    def chat_revise(self,
                    content_type: str,
                    current_content: str,
                    user_message: str,
                    chat_history: list | None = None,
                    llm: LLM | str = models["gemini-2.0-flash"]) -> str:
        """
        通过对话方式修改已生成的内容（想法、方法等）。

        Args:
            content_type: 内容类型名称，用于提示词，如 '研究想法'、'研究方法'
            current_content: 当前内容文本
            user_message: 用户本轮的修改要求
            chat_history: 历史对话列表 [{"role": "user"|"assistant", "content": "..."}]
            llm: 使用的 LLM 模型

        Returns:
            修改后的完整内容
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        llm_obj = _llm_parser(llm)
        model = build_llm(model=llm_obj.name, temperature=0.7, key_manager=self.keys)

        messages = [
            SystemMessage(content=(
                f"你是一位专业的科研写作助手，帮助用户修改他们的{content_type}。\n"
                f"用户会提出修改要求，你须输出**完整的修改后内容**。\n"
                f"直接输出修改后的内容，不要添加「以下是修改后内容」等多余前缀。\n"
                f"保留原有优秀部分，只修改用户指出的问题。"
            )),
        ]

        # 第一轮：把当前内容和修改要求一起发送
        if not chat_history:
            messages.append(HumanMessage(
                content=f"当前{content_type}：\n\n{current_content}\n\n---\n修改要求：{user_message}"
            ))
        else:
            # 已有历史：第一轮包含原始内容，后续只追加对话
            first_user = chat_history[0] if chat_history else None
            if first_user and first_user["role"] == "user":
                messages.append(HumanMessage(
                    content=f"当前{content_type}：\n\n{current_content}\n\n---\n修改要求：{first_user['content']}"
                ))
                for msg in chat_history[1:]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=user_message))

        response = model.invoke(messages)
        return response.content

    def _build_experiment_input(self, llm: LLM, max_inner_iter: int,
                                max_outer_iter: int, timeout: int, verbose: bool) -> dict:
        """构建实验图的初始输入状态。"""
        f_desc    = os.path.join(self.project_dir, INPUT_FILES, DESCRIPTION_FILE)
        f_methods = os.path.join(self.project_dir, INPUT_FILES, METHOD_FILE)
        plots_dir = os.path.join(self.project_dir, INPUT_FILES, PLOTS_FOLDER)
        return {
            "files": {
                "Folder":           self.project_dir,
                "data_description": f_desc,
                "methods":          f_methods,
                "plots_dir":        plots_dir,
            },
            "llm": {
                "model":             llm.name,
                "temperature":       llm.temperature,
                "max_output_tokens": llm.max_output_tokens,
                "stream_verbose":    verbose,
            },
            "keys":           self.keys,
            "max_inner_iter": max_inner_iter,
            "max_outer_iter": max_outer_iter,
            "timeout":        timeout,
        }

    def run_experiment_start(self,
                             llm: LLM | str = models["gemini-2.0-flash"],
                             max_inner_iter: int = 3,
                             max_outer_iter: int = DEFAULT_EXP_MAX_ITER,
                             timeout: int = DEFAULT_EXEC_TIMEOUT,
                             verbose: bool = False,
                             thread_id: str = "exp-1") -> dict:
        """
        启动自动实验（支持 Human-in-the-Loop）。

        遇到两个介入点时图会暂停并返回 interrupt 信息，需调用
        run_experiment_resume() 提供用户输入后继续。

        Returns:
            {
              "status":    "waiting_criteria" | "waiting_lit_review" | "done" | "error",
              "data":      interrupt 数据 或 最终结果摘要,
              "thread_id": thread_id,
            }
        """
        from .experiment_agents.agents_graph import build_experiment_graph
        from langgraph.types import Command

        self._backup_file(METHOD_FILE)
        llm_obj = _llm_parser(llm)
        graph   = build_experiment_graph()
        config  = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
        input_state = self._build_experiment_input(
            llm_obj, max_inner_iter, max_outer_iter, timeout, verbose)

        self._exp_graphs[thread_id] = graph
        return self._invoke_and_check(graph, input_state, config, thread_id)

    def run_experiment_resume(self,
                              thread_id: str,
                              user_input) -> dict:
        """
        在 interrupt 点提供用户输入后继续实验。

        Args:
            thread_id:  与 run_experiment_start() 相同的 thread_id
            user_input: 针对 criteria_confirm → 字符串（确认后的标准）
                        针对 lit_review      → {"decision": "continue|stop|modify", "input": "..."}

        Returns:
            同 run_experiment_start() 的 status dict
        """
        from langgraph.types import Command

        graph  = self._exp_graphs.get(thread_id)
        if graph is None:
            return {"status": "error", "data": "未找到实验会话，请重新启动。", "thread_id": thread_id}
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
        return self._invoke_and_check(graph, Command(resume=user_input), config, thread_id)

    def _invoke_and_check(self, graph, input_or_command, config: dict, thread_id: str) -> dict:
        """执行图调用并将结果转换为统一 status dict。"""
        from langgraph.errors import GraphInterrupt

        try:
            result = graph.invoke(input_or_command, config)
        except GraphInterrupt as gi:
            # 旧版 LangGraph 可能抛出异常而非返回
            interrupt_val = gi.args[0][0].value if gi.args else {}
            return self._interrupt_to_status(interrupt_val, thread_id)
        except Exception as e:
            logger.error("实验执行异常：%s", e)
            return {"status": "error", "data": str(e), "thread_id": thread_id}

        # 新版 LangGraph：interrupt 后 invoke 直接返回当前 state snapshot
        # 检查是否有 pending interrupt
        try:
            state_snap = graph.get_state(config)
            pending    = state_snap.tasks  # 有 interrupt 时会有 pending tasks
            if pending:
                for task in pending:
                    interrupts = getattr(task, "interrupts", [])
                    if interrupts:
                        return self._interrupt_to_status(interrupts[0].value, thread_id)
        except Exception:
            pass

        # 没有 interrupt → 实验完成
        summary = result.get("result_summary", "") if isinstance(result, dict) else ""
        try:
            self.set_results()
        except FileNotFoundError:
            pass
        # 清理 graph 引用
        self._exp_graphs.pop(thread_id, None)
        return {"status": "done", "data": summary, "thread_id": thread_id}

    # -----------------------------------------------------------------------
    # 后台线程实验（供 Streamlit 调用）
    # -----------------------------------------------------------------------

    def run_experiment_bg_start(self,
                                llm: "LLM | str" = None,
                                max_inner_iter: int = 3,
                                max_outer_iter: int = None,
                                timeout: int = None,
                                verbose: bool = False,
                                thread_id: str = "exp-bg") -> str:
        """
        在后台线程中启动实验，立即返回 module_folder 路径。
        Streamlit 通过轮询 {module_folder}/bg_status.json 获取状态。

        Returns:
            module_folder: 实验输出目录路径
        """
        from .experiment_agents.agents_graph import build_experiment_graph

        if llm is None:
            llm = models["gemini-2.0-flash"]
        if max_outer_iter is None:
            max_outer_iter = DEFAULT_EXP_MAX_ITER
        if timeout is None:
            timeout = DEFAULT_EXEC_TIMEOUT

        module_folder = os.path.join(self.project_dir, "experiment_output")
        os.makedirs(module_folder, exist_ok=True)

        # 清除上轮信号文件
        for fname in ("bg_status.json", "bg_resume.json"):
            fp = os.path.join(module_folder, fname)
            if os.path.exists(fp):
                os.remove(fp)

        self._backup_file(METHOD_FILE)
        llm_obj = _llm_parser(llm)
        graph = build_experiment_graph()
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
        input_state = self._build_experiment_input(
            llm_obj, max_inner_iter, max_outer_iter, timeout, verbose)

        self._exp_graphs[thread_id] = graph

        t = threading.Thread(
            target=self._bg_worker,
            args=(graph, input_state, config, thread_id, module_folder),
            daemon=True,
        )
        t.start()
        return module_folder

    def _bg_worker(self, graph, input_or_cmd, config: dict,
                   thread_id: str, module_folder: str) -> None:
        """后台线程：执行图，写 bg_status.json，在 interrupt 点等待恢复信号。"""
        from langgraph.errors import GraphInterrupt

        status_file = os.path.join(module_folder, "bg_status.json")

        def _write(status: str, data=None):
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"status": status, "data": data or {}}, f, ensure_ascii=False)

        _write("running")
        result = None
        try:
            result = graph.invoke(input_or_cmd, config)
        except GraphInterrupt as gi:
            # 旧版 LangGraph：interrupt() 抛 GraphInterrupt
            try:
                interrupt_val = gi.args[0][0].value if gi.args else {}
            except (IndexError, TypeError):
                interrupt_val = gi.args[0] if gi.args else {}
            logger.info("捕获 GraphInterrupt：%s", interrupt_val)
            _write(self._bg_interrupt_type(interrupt_val), interrupt_val)
            self._bg_wait_resume(graph, config, thread_id, module_folder)
            return
        except Exception as e:
            logger.error("实验后台线程异常：%s", e)
            _write("error", {"message": str(e)})
            return

        # 新版 LangGraph：invoke() 正常返回，interrupt 挂起在 state 中
        try:
            state_snap = graph.get_state(config)
            # 方式1：tasks[i].interrupts（新版）
            for task in (state_snap.tasks or []):
                for intr in getattr(task, "interrupts", []):
                    iv = intr.value if hasattr(intr, "value") else intr
                    logger.info("从 state.tasks 检测到 interrupt：%s", iv)
                    _write(self._bg_interrupt_type(iv), iv)
                    self._bg_wait_resume(graph, config, thread_id, module_folder)
                    return
            # 方式2：state_snap.next 非空说明图未完成（仍有节点等待执行）
            if getattr(state_snap, "next", None):
                logger.info("graph.get_state().next 非空，图已暂停：%s", state_snap.next)
                # 尝试从最新 state values 里获取 interrupt payload
                vals = state_snap.values or {}
                last_intr = vals.get("__interrupt__") or vals.get("interrupt_value")
                if last_intr and isinstance(last_intr, dict):
                    _write(self._bg_interrupt_type(last_intr), last_intr)
                else:
                    _write("waiting_unknown", {})
                self._bg_wait_resume(graph, config, thread_id, module_folder)
                return
        except Exception as snap_err:
            logger.warning("get_state 检测 interrupt 失败：%s", snap_err)

        # 完成
        summary = result.get("result_summary", "") if isinstance(result, dict) else ""
        _write("done", {"summary": summary})
        try:
            self.set_results()
        except FileNotFoundError:
            pass
        self._exp_graphs.pop(thread_id, None)

    def _bg_wait_resume(self, graph, config: dict,
                        thread_id: str, module_folder: str) -> None:
        """轮询 bg_resume.json，收到后继续执行图。"""
        from langgraph.types import Command

        resume_file = os.path.join(module_folder, "bg_resume.json")
        while True:
            if os.path.exists(resume_file):
                try:
                    with open(resume_file, "r", encoding="utf-8") as f:
                        resume_data = json.load(f)
                    os.remove(resume_file)
                except Exception:
                    time.sleep(0.2)
                    continue
                self._bg_worker(
                    graph, Command(resume=resume_data["value"]),
                    config, thread_id, module_folder,
                )
                return
            time.sleep(0.3)

    def run_experiment_bg_resume(self, module_folder: str, user_input) -> None:
        """
        向后台线程发送恢复信号。

        Args:
            module_folder: run_experiment_bg_start() 返回的目录
            user_input:    criteria_confirm → str；lit_review → {"decision":..., "input":...}
        """
        resume_file = os.path.join(module_folder, "bg_resume.json")
        with open(resume_file, "w", encoding="utf-8") as f:
            json.dump({"value": user_input}, f, ensure_ascii=False)

    @staticmethod
    def _bg_interrupt_type(interrupt_val: dict) -> str:
        itype = interrupt_val.get("type", "")
        if itype == "dataset_select":
            return "waiting_dataset"
        if itype == "criteria_confirm":
            return "waiting_criteria"
        if itype == "lit_review":
            return "waiting_lit_review"
        return "waiting_unknown"

    @staticmethod
    def _interrupt_to_status(interrupt_val: dict, thread_id: str) -> dict:
        itype = interrupt_val.get("type", "")
        if itype == "dataset_select":
            return {"status": "waiting_dataset", "data": interrupt_val, "thread_id": thread_id}
        if itype == "criteria_confirm":
            return {"status": "waiting_criteria", "data": interrupt_val, "thread_id": thread_id}
        if itype == "lit_review":
            return {"status": "waiting_lit_review", "data": interrupt_val, "thread_id": thread_id}
        return {"status": "waiting_unknown", "data": interrupt_val, "thread_id": thread_id}

    def run_experiment(self,
                       llm: LLM | str = models["gemini-2.0-flash"],
                       max_outer_iter: int = DEFAULT_EXP_MAX_ITER,
                       timeout: int = DEFAULT_EXEC_TIMEOUT,
                       verbose: bool = False) -> str:
        """
        自动实验（无 HITL，自动接受 LLM 推断的验收标准，结果不足时自动继续）。
        适用于 API 调用、Jupyter Notebook 场景。

        Returns:
            最终实验结果摘要文本
        """
        from .experiment_agents.agents_graph import build_experiment_graph
        from langgraph.types import Command

        self._backup_file(METHOD_FILE)
        llm_obj = _llm_parser(llm)
        graph   = build_experiment_graph()
        config  = {"configurable": {"thread_id": "exp-auto"}, "recursion_limit": 200}
        input_state = self._build_experiment_input(llm_obj, 3, max_outer_iter, timeout, verbose)

        logger.info("启动自动实验（无 HITL，最多 %d 外层轮，超时 %ds）...", max_outer_iter, timeout)
        start_time = time.time()

        # 第一次调用 → 遇到 criteria_confirm interrupt
        result = graph.invoke(input_state, config)
        result = self._auto_resume_interrupts(graph, result, config)

        elapsed = time.time() - start_time
        logger.info("自动实验完成，用时 %d 分 %d 秒。", int(elapsed // 60), int(elapsed % 60))
        try:
            self.set_results()
        except FileNotFoundError:
            pass
        return result.get("result_summary", "") if isinstance(result, dict) else ""

    def _auto_resume_interrupts(self, graph, result, config: dict):
        """自动通过所有 interrupt 点（使用默认值），用于无 HITL 模式。"""
        from langgraph.types import Command

        for _ in range(20):  # 最多循环 20 次防止死循环
            try:
                state_snap = graph.get_state(config)
                pending    = state_snap.tasks
            except Exception:
                break
            has_interrupt = False
            for task in pending:
                interrupts = getattr(task, "interrupts", [])
                if interrupts:
                    has_interrupt = True
                    val   = interrupts[0].value
                    itype = val.get("type", "")
                    if itype == "criteria_confirm":
                        # 自动接受 LLM 推断的标准
                        resume_val = val.get("criteria", "")
                    elif itype == "lit_review":
                        # 自动继续优化
                        resume_val = {"decision": "continue", "input": ""}
                    else:
                        resume_val = ""
                    result = graph.invoke(Command(resume=resume_val), config)
                    break
            if not has_interrupt:
                break
        return result

    def full_pipeline(self, data_description: str | None = None) -> None:
        """
        一键运行完整流程：
        set_data_description → get_idea → get_method → run_experiment → get_paper
        """
        self.set_data_description(data_description)
        self.get_idea()
        self.get_method()
        self.run_experiment()
        self.get_paper()
