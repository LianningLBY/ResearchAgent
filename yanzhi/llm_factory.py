"""
LLM 初始化工厂函数。

两处 preprocess_node（langgraph_agents 和 paper_agents）原本各自
重复实现相同的模型判断逻辑，统一到此处。
"""

from .key_manager import KeyManager
from .log import get_logger

logger = get_logger(__name__)


def build_llm(model: str, temperature: float, key_manager: KeyManager):
    """
    根据模型名称实例化对应的 LangChain Chat 模型。

    Args:
        model: 模型名称字符串（如 'gemini-2.0-flash'）
        temperature: 采样温度
        key_manager: API key 管理器

    Returns:
        LangChain BaseChatModel 实例

    Raises:
        ValueError: 不支持的模型名称
    """
    model_lower = model.lower()

    if "gemini" in model_lower:
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.debug("初始化 Gemini 模型：%s", model)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=key_manager.GEMINI,
        )

    if any(k in model_lower for k in ("gpt", "o3", "o1")):
        from langchain_openai import ChatOpenAI
        logger.debug("初始化 OpenAI 模型：%s", model)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=key_manager.OPENAI,
        )

    if "claude" in model_lower or "anthropic" in model_lower:
        from langchain_anthropic import ChatAnthropic
        logger.debug("初始化 Anthropic 模型：%s", model)
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=key_manager.ANTHROPIC,
        )

    if "minimax" in model_lower or "abab" in model_lower:
        from langchain_openai import ChatOpenAI
        logger.debug("初始化 MiniMax 模型（OpenAI 兼容）：%s", model)
        return ChatOpenAI(
            model=model,
            api_key=key_manager.MINIMAX,
            temperature=temperature,
            base_url="https://api.minimax.chat/v1",
        )

    raise ValueError(
        f"不支持的模型：'{model}'。"
        f"请检查模型名称是否包含 gemini / gpt / o3 / claude / minimax 等关键词。"
    )
