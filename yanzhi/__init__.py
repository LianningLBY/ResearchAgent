"""
研智 (YanZhi) — 中文多智能体科研助手
======================================

基于 LangGraph 构建的中文版多智能体科研辅助系统。
全流程使用中文提示词，支持 ctex 中文 LaTeX 排版。

快速开始::

    from yanzhi import YanZhi, Journal

    yz = YanZhi(project_dir="我的项目")

    yz.set_data_description(\"\"\"
    我们有一批气候模型输出数据，包含温度、降水、风速等变量的时间序列。
    请分析不同气候模式下的变量相关性，并探讨极端天气事件的统计规律。
    \"\"\")

    yz.get_idea(llm='gemini-2.0-flash', iterations=4)
    yz.check_idea(max_iterations=7)
    yz.get_method(llm='gemini-2.0-flash')
    yz.get_paper(journal=Journal.CNKI)
    yz.referee()
"""

from .main import YanZhi
from .paper_agents.journal import Journal
from .key_manager import KeyManager
from .llm import LLM, models
from .research import Research
from .log import get_logger, setup_logging
from .llm_factory import build_llm

__all__ = ["YanZhi", "Journal", "KeyManager", "LLM", "models", "Research",
           "get_logger", "setup_logging", "build_llm"]
__version__ = "1.0.0"
