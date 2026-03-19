from langchain_core.runnables import RunnableConfig

from ..paper_agents.tools import extract_latex_block, LLM_call_stream, clean_section
from .prompts import reviewer_fast_prompt
from .parameters import GraphState
from .pdf_reader import pdf_to_images


def referee(state: GraphState, config: RunnableConfig):
    print('正在审阅论文...', end="", flush=True)

    paper_name = ("paper_v2_no_citations.pdf" if state['referee']['paper_version'] == 2
                  else "paper_v4_final.pdf")
    pdf_path = f"{state['files']['Paper_folder']}/{paper_name}"
    out_dir  = state['files']['paper_images']

    state['referee']['images'] = pdf_to_images(pdf_path, out_dir)

    PROMPT = reviewer_fast_prompt(state)
    state, result = LLM_call_stream(PROMPT, state)
    text = extract_latex_block(state, result, "REVIEW")
    text = clean_section(text, "REVIEW")

    with open(state['files']['referee_report'], 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"完成 {state['tokens']['ti']} {state['tokens']['to']}")
