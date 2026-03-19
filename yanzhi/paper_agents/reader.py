import os
import time
import hashlib
import shutil
from pathlib import Path
from langchain_core.runnables import RunnableConfig

from .parameters import GraphState
from .latex_presets import journal_dict
from ..config import (INPUT_FILES, IDEA_FILE, METHOD_FILE, RESULTS_FILE,
                      PAPER_FOLDER, PLOTS_FOLDER, PAPER_V1, PAPER_V2, PAPER_V3, PAPER_V4)
from ..llm_factory import build_llm
from ..log import get_logger

logger = get_logger(__name__)


def preprocess_node(state: GraphState, config: RunnableConfig):
    """读取输入文件、初始化 LLM、设置文件路径"""

    # 初始化 LLM（统一通过 llm_factory 创建）
    state['llm']['llm'] = build_llm(
        model=state['llm']['model'],
        temperature=state['llm']['temperature'],
        key_manager=state['keys'],
    )

    state['tokens'] = {'ti': 0, 'to': 0, 'i': 0, 'o': 0}
    state['time']   = {'start': time.time()}
    state['params'] = {'num_keywords': 5}

    state['files'] = {**state['files'],
                      "Paper_folder": f"{state['files']['Folder']}/{PAPER_FOLDER}"}
    os.makedirs(state['files']['Paper_folder'], exist_ok=True)

    state['files'] = {**state['files'],
                      "Idea":      IDEA_FILE,
                      "Methods":   METHOD_FILE,
                      "Results":   RESULTS_FILE,
                      "Plots":     PLOTS_FOLDER,
                      "Paper_v1":  PAPER_V1,
                      "Paper_v2":  PAPER_V2,
                      "Paper_v3":  PAPER_V3,
                      "Paper_v4":  PAPER_V4,
                      "Error":     f"{state['files']['Paper_folder']}/Error.txt",
                      "LaTeX_log": f"{state['files']['Paper_folder']}/LaTeX_compilation.log",
                      "LaTeX_err": f"{state['files']['Paper_folder']}/LaTeX_err.log",
                      "Temp":      f"{state['files']['Paper_folder']}/temp",
                      "LLM_calls": f"{state['files']['Paper_folder']}/LLM_calls.txt"}

    state['latex']  = {'section': ''}

    idea = {}
    for key in ["Idea", "Methods", "Results"]:
        path = Path(f"{state['files']['Folder']}/{INPUT_FILES}/{state['files'][key]}")
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                idea[key] = f.read()
        else:
            idea[key] = None

    # 清理旧论文文件
    for f in ['Paper_v1', 'Paper_v2', 'Paper_v3', 'Paper_v4']:
        f_in = f"{state['files']['Paper_folder']}/{state['files'][f]}"
        if os.path.exists(f_in):
            os.remove(f_in)
        root = Path(state['files'][f]).stem
        for fname in [f'{root}.pdf', f'{root}.aux', f'{root}.log', f'{root}.out',
                      f'{root}.bbl', f'{root}.blg', f'{root}.synctex.gz',
                      'bibliography.bib', 'bibliography_temp.bib']:
            fp = f"{state['files']['Paper_folder']}/{fname}"
            if os.path.exists(fp):
                os.remove(fp)

    for f_in in [state['files']['Error'], state['files']['LLM_calls'],
                 state['files']['LaTeX_log'], state['files']['LaTeX_err']]:
        if os.path.exists(f_in):
            os.remove(f_in)

    os.makedirs(state['files']['Temp'], exist_ok=True)

    # 创建 input_files 的符号链接（用于 LaTeX 编译时引用图片）
    link_src = Path(f"{state['files']['Folder']}/{INPUT_FILES}").resolve()
    link_dst = Path(f"{state['files']['Paper_folder']}/{INPUT_FILES}").resolve()
    if not link_dst.exists() and not link_dst.is_symlink():
        link_dst.symlink_to(link_src, target_is_directory=True)

    # 处理重复图片
    plots_dir    = Path(f"{state['files']['Folder']}/{INPUT_FILES}/{state['files']['Plots']}")
    repeated_dir = Path(f"{plots_dir}_repeated")
    hash_dict = {}
    if plots_dir.exists():
        for file in plots_dir.iterdir():
            if file.is_file():
                with open(file, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in hash_dict:
                    repeated_dir.mkdir(exist_ok=True)
                    logger.info("重复图片：%s（与 %s 相同，已移至 repeated 目录）",
                                file.name, hash_dict[file_hash].name)
                    shutil.move(file, repeated_dir / file.name)
                else:
                    hash_dict[file_hash] = file

    # 统计图片数量
    folder_path = Path(f"{state['files']['Folder']}/{INPUT_FILES}/{state['files']['Plots']}")
    if folder_path.exists():
        files = [f for f in folder_path.iterdir() if f.is_file() and f.name != '.DS_Store']
        state['files']['num_plots'] = len(files)
    else:
        state['files']['num_plots'] = 0

    return {**state,
            "llm": state['llm'], "tokens": state['tokens'], "params": state['params'],
            "files": state['files'], "latex": state['latex'],
            "idea": idea, "paper": {**state['paper'], "summary": ""},
            "time": state['time']}
