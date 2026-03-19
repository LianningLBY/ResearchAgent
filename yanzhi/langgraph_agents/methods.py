from langchain_core.runnables import RunnableConfig

from ..paper_agents.tools import extract_latex_block, LLM_call_stream, clean_section
from .prompts import methods_fast_prompt
from .parameters import GraphState


def methods_fast(state: GraphState, config: RunnableConfig):
    print('正在生成研究方法...', end="", flush=True)

    PROMPT = methods_fast_prompt(state)
    state, result = LLM_call_stream(PROMPT, state)
    text = extract_latex_block(state, result, "METHODS")
    text = clean_section(text, "METHODS")

    with open(state['files']['methods'], 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"完成 {state['tokens']['ti']} {state['tokens']['to']}")
