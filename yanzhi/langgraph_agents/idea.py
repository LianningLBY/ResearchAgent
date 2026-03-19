from langchain_core.runnables import RunnableConfig

from ..paper_agents.tools import extract_latex_block, LLM_call_stream, clean_section
from .prompts import idea_maker_prompt, idea_hater_prompt
from .parameters import GraphState


def idea_maker(state: GraphState, config: RunnableConfig):
    print(f"想法生成器（第 {state['idea']['iteration']+1} 轮）")

    PROMPT = idea_maker_prompt(state)
    state, result = LLM_call_stream(PROMPT, state)
    text = extract_latex_block(state, result, "IDEA")
    text = clean_section(text, "IDEA")

    state['idea']['idea'] = text
    state['idea']['previous_ideas'] = (
        f"{state['idea']['previous_ideas']}\n\n"
        f"第 {state['idea']['iteration']} 轮：\n想法：{text}\n"
    )
    state['idea']['iteration'] += 1

    if state['idea']['iteration'] == state['idea']['total_iterations']:
        with open(state['files']['idea'], 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"完成 {state['tokens']['ti']} {state['tokens']['to']}")

    return {"idea": state['idea']}


def idea_hater(state: GraphState, config: RunnableConfig):
    print(f"想法批评者（第 {state['idea']['iteration']} 轮）")

    PROMPT = idea_hater_prompt(state)
    state, result = LLM_call_stream(PROMPT, state)
    text = extract_latex_block(state, result, "CRITIC")
    text = clean_section(text, "CRITIC")

    state['idea']['criticism'] = text
    return {"idea": state['idea']}
