import os
from langchain_core.runnables import RunnableConfig

from .parameters import GraphState
from ..config import INPUT_FILES, IDEA_FILE, METHOD_FILE, LITERATURE_FILE, REFEREE_FILE, PAPER_FOLDER
from ..llm_factory import build_llm
from ..log import get_logger

logger = get_logger(__name__)


def preprocess_node(state: GraphState, config: RunnableConfig):
    """初始化 LLM、读取输入文件、设置文件路径"""

    state['tokens'] = {'ti': 0, 'to': 0, 'i': 0, 'o': 0}

    # 初始化 LLM（统一通过 llm_factory 创建）
    import sys as _sys
    _km = state['keys']
    print(f"[DEBUG preprocess] model={state['llm']['model']} GEMINI_key={'已设置' if getattr(_km,'GEMINI',None) else '空/None'}", file=_sys.stderr, flush=True)
    state['llm']['llm'] = build_llm(
        model=state['llm']['model'],
        temperature=state['llm']['temperature'],
        key_manager=state['keys'],
    )

    # 读取数据描述
    try:
        with open(state['files']['data_description'], 'r', encoding='utf-8') as f:
            description = f.read()
    except FileNotFoundError:
        raise FileNotFoundError("未找到数据描述文件，请先调用 set_data_description()。")

    # 读取想法文件（方法生成/文献检索时需要）
    if state['task'] in ['methods_generation', 'literature']:
        try:
            with open(state['files']['idea'], 'r', encoding='utf-8') as f:
                idea = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("未找到想法文件，请先调用 get_idea() 或 set_idea()。")

    # 设置模块输出文件夹
    task_map = {
        'idea_generation':    'idea_generation_output',
        'methods_generation': 'methods_generation_output',
        'literature':         'literature_output',
        'referee':            'referee_output',
    }
    module_folder = task_map[state['task']]
    state['files']['module_folder'] = module_folder
    state['files']['f_stream'] = f"{state['files']['Folder']}/{module_folder}/{state['task']}.log"

    state['files'] = {**state['files'],
                      "Temp":      f"{state['files']['Folder']}/{module_folder}",
                      "LLM_calls": f"{state['files']['Folder']}/{module_folder}/LLM_calls.txt",
                      "Error":     f"{state['files']['Folder']}/{module_folder}/Error.txt"}

    literature_text = ""
    if state['task'] == 'idea_generation':
        lit_path = state['files'].get('literature')
        if lit_path and os.path.exists(lit_path):
            try:
                with open(lit_path, 'r', encoding='utf-8') as f:
                    literature_text = f.read()
            except Exception:
                literature_text = ""
        idea = {**state['idea'], 'iteration': 0, 'previous_ideas': '', 'idea': '', 'criticism': ''}
        state['files'] = {**state['files'],
                          "idea":     f"{state['files']['Folder']}/{INPUT_FILES}/{IDEA_FILE}",
                          "idea_log": f"{state['files']['Folder']}/{module_folder}/idea.log"}
    elif state['task'] == 'methods_generation':
        state['files'] = {**state['files'],
                          "methods": f"{state['files']['Folder']}/{INPUT_FILES}/{METHOD_FILE}"}
        idea = {**state['idea'], 'idea': idea}
    elif state['task'] == 'literature':
        state['literature'] = {**state['literature'], 'iteration': 0, 'query': '', 'decision': '',
                                'papers': '', 'next_agent': '', 'messages': '', 'num_papers': 0}
        state['files'] = {**state['files'],
                          "literature":     f"{state['files']['Folder']}/{INPUT_FILES}/{LITERATURE_FILE}",
                          "literature_log": f"{state['files']['Folder']}/{module_folder}/literature.log",
                          "papers":         f"{state['files']['Folder']}/{module_folder}/papers_processed.log"}
        idea = {**state['idea'], 'idea': idea}
    elif state['task'] == 'referee':
        state['referee'] = {**state['referee'], 'paper_version': 2, 'report': '', 'images': []}
        state['files'] = {**state['files'],
                          "Paper_folder":   f"{state['files']['Folder']}/{PAPER_FOLDER}",
                          "referee_report": f"{state['files']['Folder']}/{INPUT_FILES}/{REFEREE_FILE}",
                          "referee_log":    f"{state['files']['Folder']}/{module_folder}/referee.log",
                          "paper_images":   f"{state['files']['Folder']}/{module_folder}"}

    # 创建目录
    os.makedirs(state['files']['Folder'], exist_ok=True)
    os.makedirs(state['files']['Temp'], exist_ok=True)
    os.makedirs(f"{state['files']['Folder']}/{INPUT_FILES}", exist_ok=True)

    # 清理旧文件
    for f in ["LLM_calls", "Error"]:
        fp = state['files'][f]
        if os.path.exists(fp):
            os.remove(fp)

    if state['task'] == 'idea_generation':
        for f in ["idea", "idea_log"]:
            fp = state['files'].get(f)
            if fp and os.path.exists(fp):
                os.remove(fp)

    if state['task'] == 'methods_generation':
        fp = state['files'].get("methods")
        if fp and os.path.exists(fp):
            os.remove(fp)

    if state['task'] == 'literature':
        for f in ['literature', 'literature_log', 'papers']:
            fp = state['files'].get(f)
            if fp and os.path.exists(fp):
                os.remove(fp)

    if state['task'] == 'referee':
        for f in ['referee_report', 'referee_log']:
            fp = state['files'].get(f)
            if fp and os.path.exists(fp):
                os.remove(fp)
        return {**state, "files": state['files'], "llm": state['llm'],
                "tokens": state['tokens'], "data_description": description,
                "referee": state['referee'], "literature_text": literature_text}

    return {**state, "files": state['files'], "llm": state['llm'],
            "tokens": state['tokens'], "data_description": description, "idea": idea,
            "literature_text": literature_text}
