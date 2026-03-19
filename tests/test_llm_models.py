"""测试 llm.py 中的模型注册表。"""

import pytest
from yanzhi.llm import LLM, models


def test_models_registry_not_empty():
    assert len(models) > 0


def test_all_models_have_required_fields():
    for name, model in models.items():
        assert isinstance(model, LLM), f"{name} 应为 LLM 实例"
        assert model.name, f"{name} 应有非空 name"
        assert model.max_output_tokens > 0, f"{name}.max_output_tokens 应为正数"
        assert model.temperature is None or 0.0 <= model.temperature <= 2.0, (
            f"{name}.temperature 超出合理范围"
        )


def test_default_model_exists():
    assert "gemini-2.0-flash" in models


def test_model_names_consistent():
    """models 字典的 key 与 LLM.name 不必完全一致（有别名），但值的类型要正确。"""
    for key, model in models.items():
        assert isinstance(model.name, str)
        assert len(model.name) > 0
