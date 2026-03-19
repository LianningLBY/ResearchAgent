"""
研智日志模块。

使用方式：
    from yanzhi.log import get_logger
    logger = get_logger(__name__)
    logger.info("正在生成研究想法...")
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_initialized = False


def setup_logging(level: int = logging.INFO) -> None:
    """初始化全局日志配置（幂等，重复调用无副效应）。"""
    global _initialized
    if _initialized:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger("yanzhi")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger，自动完成初始化。"""
    setup_logging()
    return logging.getLogger(name)
