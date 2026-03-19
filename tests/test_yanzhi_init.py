"""测试 YanZhi 类的初始化和基础方法（无需 API Key）。"""

import os
import tempfile
import pytest
from yanzhi import YanZhi


@pytest.fixture
def yz(tmp_path):
    """创建临时项目目录的 YanZhi 实例。"""
    return YanZhi(project_dir=str(tmp_path / "test_project"))


def test_project_dir_created(yz, tmp_path):
    assert os.path.isdir(yz.project_dir)


def test_input_files_dir_created(yz):
    from yanzhi.config import INPUT_FILES
    assert os.path.isdir(os.path.join(yz.project_dir, INPUT_FILES))


def test_plots_folder_created(yz):
    assert os.path.isdir(yz.plots_folder)


def test_set_data_description(yz):
    text = "这是测试用的研究方向描述。"
    yz.set_data_description(text)
    assert yz.research.data_description == text


def test_set_idea(yz):
    text = "# 研究想法\n使用深度学习方法。"
    yz.set_idea(text)
    assert yz.research.idea == text


def test_set_method(yz):
    text = "## 方法\n1. 数据预处理"
    yz.set_method(text)
    assert yz.research.methodology == text


def test_set_results(yz):
    text = "实验结果：准确率 95%"
    yz.set_results(text)
    assert yz.research.results == text


def test_backup_creates_history_file(yz):
    from yanzhi.config import IDEA_FILE, INPUT_FILES
    yz.set_idea("初始想法")
    backup = yz._backup_file(IDEA_FILE)
    assert backup is not None
    assert os.path.exists(backup)


def test_list_history_empty_initially(yz):
    result = yz.list_history("idea")
    assert isinstance(result, list)
    assert len(result) == 0


def test_list_history_after_backup(yz):
    import time
    yz.set_idea("版本一")
    yz._backup_file("idea.md")
    time.sleep(1)  # 确保时间戳不同
    yz.set_idea("版本二")
    yz._backup_file("idea.md")
    history = yz.list_history("idea")
    assert len(history) == 2
    # 按时间倒序，第一个是最新的
    for item in history:
        assert "path" in item
        assert "timestamp" in item
        assert "preview" in item


def test_set_all_does_not_raise(yz):
    """set_all() 在文件不存在时不应抛出异常。"""
    yz.set_all()  # 不应抛出


def test_clear_project_dir(tmp_path):
    proj = str(tmp_path / "clear_test")
    yz = YanZhi(project_dir=proj)
    yz.set_idea("先写一些内容")
    yz2 = YanZhi(project_dir=proj, clear_project_dir=True)
    assert yz2.research.idea == ""
