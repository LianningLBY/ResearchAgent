"""测试 paper_agents/tools.py 中的 JSON 解析函数。"""

import pytest
from yanzhi.paper_agents.tools import json_parser2, json_parser3


VALID_JSON_BLOCK = '```json\n{"key": "value", "num": 42}\n```'
VALID_JSON_NESTED = '```json\n{"Decision": "novel", "Reason": "新方法", "Query": "deep learning"}\n```'
INVALID_JSON_BLOCK = "这里没有 JSON 代码块，只有普通文本。"
BARE_JSON = '{"Decision": "novel", "Reason": "新方法", "Query": "test"}'


class TestJsonParser2:
    def test_parse_valid_json(self):
        result = json_parser2(VALID_JSON_BLOCK)
        assert result == {"key": "value", "num": 42}

    def test_raises_on_no_block(self):
        with pytest.raises(ValueError, match="未找到"):
            json_parser2(INVALID_JSON_BLOCK)

    def test_parse_nested_keys(self):
        result = json_parser2(VALID_JSON_NESTED)
        assert result["Decision"] == "novel"
        assert result["Query"] == "deep learning"


class TestJsonParser3:
    def test_parse_valid_json(self):
        result = json_parser3(VALID_JSON_BLOCK)
        assert result["key"] == "value"

    def test_fallback_bare_json(self):
        """json_parser3 应能解析没有 ``` 包裹的裸 JSON。"""
        result = json_parser3(BARE_JSON)
        assert result["Decision"] == "novel"

    def test_raises_on_garbage(self):
        with pytest.raises((ValueError, Exception)):
            json_parser3("这完全不是 JSON 也不是 JSON 代码块")

    def test_case_insensitive_json_tag(self):
        text = '```JSON\n{"a": 1}\n```'
        result = json_parser3(text)
        assert result["a"] == 1
