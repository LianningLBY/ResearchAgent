"""测试日志模块。"""

import logging
import pytest
from yanzhi.log import get_logger, setup_logging


def test_get_logger_returns_logger():
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_setup_logging_idempotent():
    """多次调用 setup_logging 不应添加重复 handler。"""
    setup_logging()
    root = logging.getLogger("yanzhi")
    count_before = len(root.handlers)
    setup_logging()
    assert len(root.handlers) == count_before


def test_logger_has_correct_parent():
    logger = get_logger("yanzhi.test")
    assert logger.name == "yanzhi.test"


def test_logger_level_propagation(caplog):
    """通过 caplog 验证日志消息可被捕获。"""
    logger = get_logger("yanzhi.caplog_test")
    with caplog.at_level(logging.INFO, logger="yanzhi.caplog_test"):
        logger.info("测试消息 123")
    assert "测试消息 123" in caplog.text
