"""测试 JSONL token 记录格式。"""

import json
import os
import tempfile
import pytest


def _make_state(tmpfile: str) -> dict:
    return {
        "llm": {"model": "gemini-2.0-flash", "max_output_tokens": 8192},
        "tokens": {"ti": 0, "to": 0, "i": 0, "o": 0},
        "files": {"LLM_calls": tmpfile},
    }


def test_token_record_is_valid_jsonl():
    from yanzhi.paper_agents.tools import _write_token_record

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        fname = f.name
    try:
        state = _make_state(fname)
        state["tokens"]["ti"] = 100
        state["tokens"]["to"] = 200
        _write_token_record(state, input_tok=50, output_tok=100)

        with open(fname, "r", encoding="utf-8") as f:
            line = f.readline().strip()

        record = json.loads(line)
        assert record["model"] == "gemini-2.0-flash"
        assert record["i"] == 50
        assert record["o"] == 100
        assert record["ti"] == 100
        assert record["to"] == 200
        assert "ts" in record
    finally:
        os.unlink(fname)


def test_multiple_token_records_appended():
    from yanzhi.paper_agents.tools import _write_token_record

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        fname = f.name
    try:
        state = _make_state(fname)
        _write_token_record(state, input_tok=10, output_tok=20)
        _write_token_record(state, input_tok=30, output_tok=40)

        with open(fname, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        assert len(lines) == 2
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["i"] == 10
        assert r2["i"] == 30
    finally:
        os.unlink(fname)
