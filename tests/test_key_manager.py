"""测试 KeyManager。"""

import pytest
from yanzhi.key_manager import KeyManager


def test_key_manager_defaults():
    """初始 KeyManager 的所有 key 应为空字符串。"""
    km = KeyManager()
    assert km.ANTHROPIC == ""
    assert km.GEMINI == ""
    assert km.OPENAI == ""
    assert km.PERPLEXITY == ""
    assert km.SEMANTIC_SCHOLAR == ""


def test_key_manager_dict_access():
    """KeyManager 支持字典风格的 get/set。"""
    km = KeyManager()
    km["OPENAI"] = "test-key-123"
    assert km["OPENAI"] == "test-key-123"
    assert km.OPENAI == "test-key-123"


def test_key_manager_set_attribute():
    km = KeyManager(GEMINI="my-gemini-key")
    assert km.GEMINI == "my-gemini-key"


def test_key_manager_none_values():
    """KeyManager 应接受 None 值。"""
    km = KeyManager(ANTHROPIC=None, OPENAI=None)
    assert km.ANTHROPIC is None
    assert km.OPENAI is None
