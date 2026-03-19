from pydantic import BaseModel
from typing import Dict


class LLM(BaseModel):
    """LLM 模型定义"""
    name: str
    max_output_tokens: int
    temperature: float | None


gemini20flash  = LLM(name="gemini-2.0-flash",             max_output_tokens=8192,   temperature=0.7)
gemini25flash  = LLM(name="gemini-2.5-flash",             max_output_tokens=65536,  temperature=0.7)
gemini25pro    = LLM(name="gemini-2.5-pro",               max_output_tokens=65536,  temperature=0.7)
gpt4o          = LLM(name="gpt-4o-2024-11-20",            max_output_tokens=16384,  temperature=0.5)
gpt41          = LLM(name="gpt-4.1-2025-04-14",           max_output_tokens=16384,  temperature=0.5)
gpt41mini      = LLM(name="gpt-4.1-mini",                 max_output_tokens=16384,  temperature=0.5)
gpt4omini      = LLM(name="gpt-4o-mini-2024-07-18",       max_output_tokens=16384,  temperature=0.5)
claude37sonnet = LLM(name="claude-3-7-sonnet-20250219",   max_output_tokens=64000,  temperature=0)
claude4opus    = LLM(name="claude-opus-4-20250514",        max_output_tokens=32000,  temperature=0)
minimax52      = LLM(name="MiniMax-M2.1",                  max_output_tokens=32000,  temperature=0.7)
minimax25      = LLM(name="MiniMax-M2.5",                  max_output_tokens=32000,  temperature=0.7)
minimax27      = LLM(name="MiniMax-M2.7",                  max_output_tokens=32000,  temperature=0.7)
minimax_chat   = LLM(name="abab6.5-chat",                  max_output_tokens=32000,  temperature=0.7)

models: Dict[str, LLM] = {
    "gemini-2.0-flash":  gemini20flash,
    "gemini-2.5-flash":  gemini25flash,
    "gemini-2.5-pro":    gemini25pro,
    "gpt-4o":            gpt4o,
    "gpt-4.1":           gpt41,
    "gpt-4.1-mini":      gpt41mini,
    "gpt-4o-mini":       gpt4omini,
    "claude-3.7-sonnet": claude37sonnet,
    "claude-4-opus":     claude4opus,
    "MiniMax-M2.1":      minimax52,
    "MiniMax-M2.5":      minimax25,
    "MiniMax-M2.7":      minimax27,
    "abab6.5-chat":      minimax_chat,
}
