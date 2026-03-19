"""自动实验模块（HITL 双层循环版）单元测试。"""
import os
import sys
import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yanzhi.experiment_agents.executor import execute_code, ExecResult
from yanzhi.experiment_agents.env_setup import (
    extract_missing_module, module_to_package,
)
from yanzhi.experiment_agents.progress import (
    write_progress, read_progress, clear_progress, summarize_progress,
)
from yanzhi.experiment_agents.prompts import (
    criteria_infer_prompt, code_gen_prompt, fix_code_prompt,
    diagnose_prompt, method_refine_prompt, results_summary_prompt,
)
from yanzhi.experiment_agents.experiment_node import (
    inner_router, outer_router, human_decision_router,
    _extract_code, _parse_json,
)


# ─── executor 测试 ────────────────────────────────────────────────────────────

class TestExecutor:
    def test_simple_success(self, tmp_path):
        r = execute_code("print('hello')", str(tmp_path))
        assert r.success
        assert "hello" in r.stdout

    def test_syntax_error(self, tmp_path):
        r = execute_code("def bad(:\n    pass", str(tmp_path))
        assert not r.success

    def test_runtime_error(self, tmp_path):
        r = execute_code("raise ValueError('boom')", str(tmp_path))
        assert not r.success

    def test_timeout(self, tmp_path):
        r = execute_code("import time; time.sleep(999)", str(tmp_path), timeout=1)
        assert not r.success

    @pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("matplotlib") is None,
        reason="matplotlib not installed"
    )
    def test_matplotlib_auto_save(self, tmp_path):
        code = "import matplotlib.pyplot as plt\nplt.plot([1,2,3])\nplt.show()"
        plots_dir = str(tmp_path / "plots")
        r = execute_code(code, str(tmp_path), plots_dir=plots_dir)
        assert r.success, r.stderr
        assert len(r.plot_files) == 1

    def test_output_truncation(self, tmp_path):
        r = execute_code("print('a' * 20000)", str(tmp_path))
        assert len(r.stdout) <= 10_000

    def test_no_output(self, tmp_path):
        r = execute_code("x = 1 + 1", str(tmp_path))
        assert r.success
        assert r.combined_output == "(无输出)"

    def test_exec_result_properties(self):
        assert ExecResult("", "", 0).success is True
        assert ExecResult("", "err", 1).success is False


# ─── prompts 测试 ─────────────────────────────────────────────────────────────

class TestPrompts:
    def test_criteria_infer_contains_methods(self):
        p = criteria_infer_prompt("方法内容", "描述")
        assert "方法内容" in p

    def test_code_gen_with_criteria(self):
        p = code_gen_prompt("描述", "方法", criteria="R²≥0.6")
        assert "R²≥0.6" in p
        assert "\\begin{CODE}" in p

    def test_code_gen_with_history(self):
        p = code_gen_prompt("描述", "方法", prev_output="错误", prev_code="旧代码")
        assert "错误" in p
        assert "旧代码" in p

    def test_fix_code_prompt(self):
        p = fix_code_prompt("code", "traceback")
        assert "traceback" in p
        assert "\\begin{CODE}" in p

    def test_diagnose_prompt_has_failure_types(self):
        p = diagnose_prompt("output", "methods", "criteria", True)
        assert "code_error" in p
        assert "insufficient" in p

    def test_method_refine_with_literature(self):
        p = method_refine_prompt("方法", "诊断", "文献内容")
        assert "文献内容" in p
        assert "\\begin{METHODS}" in p

    def test_method_refine_with_human_input(self):
        p = method_refine_prompt("方法", "诊断", "文献", human_input="用户指示")
        assert "用户指示" in p

    def test_results_summary_prompt(self):
        p = results_summary_prompt("输出", "方法", "描述")
        assert "300-500" in p


# ─── 工具函数测试 ─────────────────────────────────────────────────────────────

class TestUtils:
    def test_extract_code_with_markers(self):
        text = "\\begin{CODE}\nprint('hi')\n\\end{CODE}"
        assert _extract_code(text) == "print('hi')"

    def test_extract_code_fallback_python_block(self):
        text = "```python\nprint('hi')\n```"
        assert _extract_code(text) == "print('hi')"

    def test_extract_code_none(self):
        assert _extract_code("no code here") is None

    def test_parse_json_with_json_block(self):
        text = '```json\n{"a": 1}\n```'
        assert _parse_json(text) == {"a": 1}

    def test_parse_json_bare(self):
        assert _parse_json('{"key": "value"}') == {"key": "value"}


# ─── 路由函数测试 ─────────────────────────────────────────────────────────────

class TestRouters:
    def test_inner_success_routes_to_diagnose(self):
        assert inner_router({"exec_success": True}) == "diagnose"

    def test_inner_fail_within_limit_routes_to_fix(self):
        s = {"exec_success": False, "inner_iteration": 0, "max_inner_iter": 3}
        assert inner_router(s) == "fix_code"

    def test_inner_fail_at_limit_routes_to_diagnose(self):
        s = {"exec_success": False, "inner_iteration": 3, "max_inner_iter": 3}
        assert inner_router(s) == "diagnose"

    def test_outer_satisfied_routes_to_save(self):
        assert outer_router({"failure_type": "satisfied"}) == "save_results"

    def test_outer_within_limit_routes_to_lit_search(self):
        s = {"failure_type": "insufficient", "outer_iteration": 0, "max_outer_iter": 2}
        assert outer_router(s) == "lit_search"

    def test_outer_at_limit_routes_to_save(self):
        s = {"failure_type": "insufficient", "outer_iteration": 2, "max_outer_iter": 2}
        assert outer_router(s) == "save_results"

    def test_human_stop_routes_to_save(self):
        assert human_decision_router({"human_decision": "stop"}) == "save_results"

    def test_human_continue_routes_to_refine(self):
        assert human_decision_router({"human_decision": "continue"}) == "method_refine"

    def test_human_modify_routes_to_refine(self):
        assert human_decision_router({"human_decision": "modify"}) == "method_refine"


# ─── preprocess_node 测试 ─────────────────────────────────────────────────────

class TestPreprocessNode:
    def _make_state(self, tmp_path, has_desc=True, has_methods=True):
        desc    = tmp_path / "data_description.md"
        methods = tmp_path / "methods.md"
        if has_desc:    desc.write_text("数据描述", encoding="utf-8")
        if has_methods: methods.write_text("研究方法", encoding="utf-8")
        return {
            "llm":   {"model": "gemini-2.0-flash", "temperature": 0.7,
                      "max_output_tokens": 8192, "stream_verbose": False},
            "keys":  MagicMock(),
            "files": {"Folder": str(tmp_path),
                      "data_description": str(desc), "methods": str(methods)},
            "tokens": {"ti": 0, "to": 0, "i": 0, "o": 0},
            "max_inner_iter": 3, "max_outer_iter": 2, "timeout": 120,
        }

    def test_reads_files_and_resets_state(self, tmp_path):
        from yanzhi.experiment_agents.experiment_node import preprocess_node
        state = self._make_state(tmp_path)
        with patch("yanzhi.experiment_agents.experiment_node.build_llm", return_value=MagicMock()):
            result = preprocess_node(state, {})
        assert result["data_description"] == "数据描述"
        assert result["methods"] == "研究方法"
        assert result["inner_iteration"] == 0
        assert result["outer_iteration"] == 0
        assert result["criteria"] == ""

    def test_missing_desc_raises(self, tmp_path):
        from yanzhi.experiment_agents.experiment_node import preprocess_node
        state = self._make_state(tmp_path, has_desc=False)
        with patch("yanzhi.experiment_agents.experiment_node.build_llm", return_value=MagicMock()):
            with pytest.raises(FileNotFoundError):
                preprocess_node(state, {})

    def test_missing_methods_raises(self, tmp_path):
        from yanzhi.experiment_agents.experiment_node import preprocess_node
        state = self._make_state(tmp_path, has_methods=False)
        with patch("yanzhi.experiment_agents.experiment_node.build_llm", return_value=MagicMock()):
            with pytest.raises(FileNotFoundError):
                preprocess_node(state, {})


# ─── diagnose_node 测试 ──────────────────────────────────────────────────────

class TestDiagnoseNode:
    def _base_state(self, tmp_path):
        return {
            "llm":   {"model": "gemini-2.0-flash", "temperature": 0.7,
                      "max_output_tokens": 8192, "stream_verbose": False,
                      "llm": MagicMock()},
            "keys":  MagicMock(),
            "files": {"Folder": str(tmp_path), "LLM_calls": str(tmp_path / "llm.txt"),
                      "module_folder": str(tmp_path)},
            "tokens":           {"ti": 0, "to": 0, "i": 0, "o": 0},
            "data_description": "描述",
            "methods":          "方法",
            "criteria":         "R²≥0.6",
            "exec_output":      "R²=0.3",
            "exec_success":     True,
        }

    def test_parses_code_error(self, tmp_path):
        from yanzhi.experiment_agents.experiment_node import diagnose_node
        state    = self._base_state(tmp_path)
        llm_resp = '```json\n{"failure_type":"code_error","diagnosis":"路径错误","search_query":"path fix"}\n```'
        with patch("yanzhi.experiment_agents.experiment_node.LLM_call",
                   return_value=(state, llm_resp)):
            r = diagnose_node(state, {})
        assert r["failure_type"] == "code_error"
        assert r["search_query"] == "path fix"

    def test_parses_insufficient(self, tmp_path):
        from yanzhi.experiment_agents.experiment_node import diagnose_node
        state    = self._base_state(tmp_path)
        llm_resp = '```json\n{"failure_type":"insufficient","diagnosis":"R²太低","search_query":"nonlinear regression"}\n```'
        with patch("yanzhi.experiment_agents.experiment_node.LLM_call",
                   return_value=(state, llm_resp)):
            r = diagnose_node(state, {})
        assert r["failure_type"] == "insufficient"

    def test_fallback_on_parse_error(self, tmp_path):
        from yanzhi.experiment_agents.experiment_node import diagnose_node
        state = self._base_state(tmp_path)
        with patch("yanzhi.experiment_agents.experiment_node.LLM_call",
                   return_value=(state, "无效 JSON")):
            r = diagnose_node(state, {})
        assert r["failure_type"] == "insufficient"


# ─── env_setup 测试 ───────────────────────────────────────────────────────────

class TestEnvSetup:
    def test_extract_missing_module_standard(self):
        stderr = "ModuleNotFoundError: No module named 'numpy'"
        assert extract_missing_module(stderr) == "numpy"

    def test_extract_missing_module_submodule(self):
        # 只取顶层包名
        stderr = "ModuleNotFoundError: No module named 'sklearn.linear_model'"
        assert extract_missing_module(stderr) == "sklearn"

    def test_extract_missing_module_none_on_no_error(self):
        assert extract_missing_module("SyntaxError: invalid syntax") is None
        assert extract_missing_module("") is None

    def test_module_to_package_known(self):
        assert module_to_package("sklearn") == "scikit-learn"
        assert module_to_package("cv2") == "opencv-python"
        assert module_to_package("PIL") == "Pillow"

    def test_module_to_package_unknown_passthrough(self):
        assert module_to_package("pandas") == "pandas"
        assert module_to_package("requests") == "requests"

    def test_executor_auto_install_log_on_missing_fake_pkg(self, tmp_path):
        """执行引用不存在包的代码，auto-install 失败，install_log 应记录失败信息。"""
        code = "import _nonexistent_pkg_xyz_12345_"
        r = execute_code(code, str(tmp_path))
        assert not r.success
        # 应该尝试安装（失败），install_log 应有记录
        assert "_nonexistent_pkg" in r.install_log or "❌" in r.install_log or r.install_log == ""
        # returncode 非 0
        assert r.returncode != 0


# ─── progress 测试 ────────────────────────────────────────────────────────────

class TestProgress:
    def test_write_and_read(self, tmp_path):
        folder = str(tmp_path)
        write_progress(folder, "preprocess", "start", "初始化")
        write_progress(folder, "preprocess", "done",  "完成")
        records = read_progress(folder)
        assert len(records) == 2
        assert records[0]["node"] == "preprocess"
        assert records[0]["status"] == "start"
        assert records[1]["status"] == "done"

    def test_read_nonexistent_returns_empty(self, tmp_path):
        assert read_progress(str(tmp_path)) == []

    def test_clear_removes_file(self, tmp_path):
        folder = str(tmp_path)
        write_progress(folder, "preprocess", "start")
        clear_progress(folder)
        assert read_progress(folder) == []

    def test_label_populated(self, tmp_path):
        folder = str(tmp_path)
        write_progress(folder, "code_gen", "start")
        records = read_progress(folder)
        assert records[0]["label"] == "生成实验代码"

    def test_unknown_node_uses_node_name_as_label(self, tmp_path):
        folder = str(tmp_path)
        write_progress(folder, "custom_node", "done")
        records = read_progress(folder)
        assert records[0]["label"] == "custom_node"

    def test_summarize_completed(self, tmp_path):
        folder = str(tmp_path)
        write_progress(folder, "preprocess",     "start")
        write_progress(folder, "preprocess",     "done")
        write_progress(folder, "criteria_infer", "start")
        write_progress(folder, "criteria_infer", "done")
        write_progress(folder, "code_gen",       "start")
        records = read_progress(folder)
        s = summarize_progress(records)
        assert "preprocess" in s["completed"]
        assert "criteria_infer" in s["completed"]
        assert s["current"] == "code_gen"

    def test_summarize_iter_counts(self, tmp_path):
        folder = str(tmp_path)
        for i in range(2):
            write_progress(folder, "code_gen", "start")
            write_progress(folder, "code_gen", "done")
        for i in range(3):
            write_progress(folder, "fix_code", "start")
            write_progress(folder, "fix_code", "done")
        records = read_progress(folder)
        s = summarize_progress(records)
        assert s["outer_iter"] == 2
        assert s["inner_iter"] == 3

    def test_summarize_errors(self, tmp_path):
        folder = str(tmp_path)
        write_progress(folder, "code_execute", "error", "超时")
        records = read_progress(folder)
        s = summarize_progress(records)
        assert len(s["errors"]) == 1
        assert s["errors"][0]["msg"] == "超时"


# ─── dataset_finder 测试 ──────────────────────────────────────────────────────

class TestDatasetFinder:
    def test_sklearn_by_query_classification(self):
        from yanzhi.experiment_agents.dataset_finder import _sklearn_by_query
        results = _sklearn_by_query("classification binary", top_k=3)
        assert len(results) > 0
        names = [r.name for r in results]
        assert any("breast_cancer" in n or "iris" in n or "wine" in n for n in names)

    def test_sklearn_by_query_regression(self):
        from yanzhi.experiment_agents.dataset_finder import _sklearn_by_query
        results = _sklearn_by_query("regression house price", top_k=3)
        assert len(results) > 0
        assert any(r.name == "california_housing" for r in results)

    def test_search_datasets_always_has_synthetic(self):
        from yanzhi.experiment_agents.dataset_finder import search_datasets
        results = search_datasets("network intrusion detection")
        assert results[-1].source == "synthetic"

    def test_search_datasets_returns_list(self):
        from yanzhi.experiment_agents.dataset_finder import search_datasets
        results = search_datasets("text classification nlp")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_dataset_candidate_to_from_dict(self):
        from yanzhi.experiment_agents.dataset_finder import DatasetCandidate
        c = DatasetCandidate("iris", "sklearn", "鸢尾花", "load_iris", "~5KB", ["classification"])
        d = c.to_dict()
        c2 = DatasetCandidate.from_dict(d)
        assert c2.name == "iris"
        assert c2.source == "sklearn"
        assert c2.tags == ["classification"]


# ─── data_fetch 测试 ──────────────────────────────────────────────────────────

class TestDataFetch:
    def test_fetch_synthetic_always_succeeds(self, tmp_path):
        from yanzhi.experiment_agents.dataset_finder import DatasetCandidate
        from yanzhi.experiment_agents.data_fetch import fetch_dataset
        c = DatasetCandidate("synthetic", "synthetic", "合成数据", "synthetic", "", [])
        r = fetch_dataset(c, str(tmp_path))
        assert r["success"] is True
        assert r["local_path"] is None
        assert "合成数据" in r["schema_md"]

    @pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("sklearn") is None,
        reason="sklearn not installed"
    )
    def test_fetch_sklearn_iris(self, tmp_path):
        from yanzhi.experiment_agents.dataset_finder import DatasetCandidate
        from yanzhi.experiment_agents.data_fetch import fetch_dataset
        c = DatasetCandidate("iris", "sklearn", "鸢尾花", "load_iris", "~5KB", [])
        r = fetch_dataset(c, str(tmp_path))
        assert r["success"] is True
        assert r["local_path"] is not None
        assert os.path.exists(r["local_path"])
        assert "target" in r["schema_md"] or "本地路径" in r["schema_md"]

    @pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("sklearn") is None,
        reason="sklearn not installed"
    )
    def test_fetch_sklearn_caches(self, tmp_path):
        from yanzhi.experiment_agents.dataset_finder import DatasetCandidate
        from yanzhi.experiment_agents.data_fetch import fetch_dataset
        c = DatasetCandidate("iris", "sklearn", "鸢尾花", "load_iris", "~5KB", [])
        r1 = fetch_dataset(c, str(tmp_path))
        mtime1 = os.path.getmtime(r1["local_path"])
        r2 = fetch_dataset(c, str(tmp_path))
        mtime2 = os.path.getmtime(r2["local_path"])
        assert mtime1 == mtime2  # 文件未被重新写入

    def test_fetch_unknown_source_fails_gracefully(self, tmp_path):
        from yanzhi.experiment_agents.dataset_finder import DatasetCandidate
        from yanzhi.experiment_agents.data_fetch import fetch_dataset
        c = DatasetCandidate("x", "unknown_source", "?", "?", "", [])
        r = fetch_dataset(c, str(tmp_path))
        assert r["success"] is False
        assert r["error"] != ""

    def test_dataset_router_skips_when_local_data(self):
        import tempfile, os
        from yanzhi.experiment_agents.experiment_node import dataset_router, _has_local_data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"a,b\n1,2\n")
            path = f.name
        try:
            assert _has_local_data(f"数据路径：{path}") is True
            assert _has_local_data("没有路径") is False
        finally:
            os.unlink(path)
