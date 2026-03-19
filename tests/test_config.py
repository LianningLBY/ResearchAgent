"""测试 config.py 中的常量定义。"""

import pytest
from yanzhi.config import (
    INPUT_FILES, PLOTS_FOLDER, PAPER_FOLDER, HISTORY_DIR,
    DESCRIPTION_FILE, IDEA_FILE, METHOD_FILE, RESULTS_FILE, LITERATURE_FILE, REFEREE_FILE,
    PAPER_V1, PAPER_V2, PAPER_V3, PAPER_V4,
    JSON_PARSE_MAX_RETRIES, SS_API_MAX_RETRIES, SS_API_RETRY_DELAY,
    LATEX_FIX_MAX_RETRIES, DEFAULT_IDEA_ITERATIONS, DEFAULT_LIT_MAX_ITER,
    HISTORY_PREVIEW_CHARS,
)


def test_file_constants_are_strings():
    """所有文件/目录常量应为非空字符串。"""
    for name, val in [
        ("INPUT_FILES", INPUT_FILES), ("PLOTS_FOLDER", PLOTS_FOLDER),
        ("PAPER_FOLDER", PAPER_FOLDER), ("HISTORY_DIR", HISTORY_DIR),
        ("DESCRIPTION_FILE", DESCRIPTION_FILE), ("IDEA_FILE", IDEA_FILE),
        ("METHOD_FILE", METHOD_FILE), ("RESULTS_FILE", RESULTS_FILE),
        ("LITERATURE_FILE", LITERATURE_FILE), ("REFEREE_FILE", REFEREE_FILE),
    ]:
        assert isinstance(val, str) and val, f"{name} 应为非空字符串"


def test_paper_version_filenames():
    """论文版本文件名应以 .tex 结尾，且包含版本号。"""
    for i, fname in enumerate([PAPER_V1, PAPER_V2, PAPER_V3, PAPER_V4], start=1):
        assert fname.endswith(".tex"), f"PAPER_V{i} 应以 .tex 结尾"
        assert f"v{i}" in fname, f"PAPER_V{i} 文件名中应包含 'v{i}'"


def test_retry_constants_are_positive():
    """重试相关常量应为正数。"""
    assert JSON_PARSE_MAX_RETRIES > 0
    assert SS_API_MAX_RETRIES > 0
    assert SS_API_RETRY_DELAY > 0
    assert LATEX_FIX_MAX_RETRIES > 0


def test_default_iterations_reasonable():
    """默认迭代轮数应在合理范围内。"""
    assert 2 <= DEFAULT_IDEA_ITERATIONS <= 20
    assert 3 <= DEFAULT_LIT_MAX_ITER <= 20


def test_history_preview_chars_positive():
    assert HISTORY_PREVIEW_CHARS > 0
