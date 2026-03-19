"""测试 main.py 中的辅助函数。"""

import os
import tempfile
import pytest
from yanzhi.main import _input_check, _llm_parser, _check_file_paths
from yanzhi.llm import models


# ── _input_check ────────────────────────────────────────────────────────────

def test_input_check_returns_string_as_is():
    assert _input_check("这是普通文本") == "这是普通文本"


def test_input_check_reads_md_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# 标题\n内容文本")
        fname = f.name
    try:
        result = _input_check(fname)
        assert "内容文本" in result
    finally:
        os.unlink(fname)


def test_input_check_non_md_returned_as_is():
    text = "/path/to/file.csv"
    assert _input_check(text) == text


# ── _llm_parser ─────────────────────────────────────────────────────────────

def test_llm_parser_string_key():
    model = _llm_parser("gemini-2.0-flash")
    assert model.name == "gemini-2.0-flash"


def test_llm_parser_model_object_passthrough():
    m = models["gemini-2.0-flash"]
    assert _llm_parser(m) is m


def test_llm_parser_invalid_key_raises():
    with pytest.raises(KeyError):
        _llm_parser("not-a-real-model")


# ── _check_file_paths ────────────────────────────────────────────────────────

def test_check_file_paths_warns_on_relative(recwarn):
    content = "- relative/path/file.csv"
    _check_file_paths(content)
    assert len(recwarn) == 1
    assert "不存在或非绝对路径" in str(recwarn[0].message)


def test_check_file_paths_no_warn_for_plain_text(recwarn):
    _check_file_paths("没有任何文件路径的纯文字内容")
    assert len(recwarn) == 0
