import random
import base64
import time
import asyncio
from functools import partial
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF

from langchain_core.runnables import RunnableConfig

from .parameters import GraphState
from .prompts import (abstract_prompt, abstract_reflection, caption_prompt,
                      clean_section_prompt, conclusions_prompt, introduction_prompt,
                      introduction_reflection, keyword_prompt, methods_prompt,
                      plot_prompt, references_prompt, refine_results_prompt, results_prompt)
from .tools import (json_parser3, LaTeX_checker, clean_section, extract_latex_block,
                    LLM_call, temp_file, check_images_in_text)
from .latex import (compile_latex, save_paper, save_bib, process_bib_file,
                    compile_tex_document, fix_latex, fix_percent)
from ..config import INPUT_FILES

# 尝试导入文献处理（可选）
try:
    from .literature import process_tex_file_with_references
    _has_citations = True
except ImportError:
    _has_citations = False


def keywords_node(state: GraphState, config: RunnableConfig):
    """提取关键词"""
    print("正在提取关键词".ljust(30, '.'), end="", flush=True)
    f_temp = Path(f"{state['files']['Temp']}/Keywords.tex")

    if f_temp.exists():
        keywords = temp_file(state, f_temp, 'read')
        print('已从 Keywords.tex 读取', end="", flush=True)
    else:
        for attempt in range(3):
            print(f'{attempt} ', end="", flush=True)
            PROMPT, _ = keyword_prompt(state)
            state, result = LLM_call(PROMPT, state)
            keywords = extract_latex_block(state, result, "Keywords")
            kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
            if len(kw_list) >= state['params']['num_keywords']:
                keywords = ', '.join(kw_list[:8])
                break
        else:
            keywords = ""
            print("关键词提取失败 ", end="", flush=True)

        temp_file(state, f_temp, 'write', keywords)
        compile_tex_document(state, f_temp, state['files']['Temp'])

    minutes, seconds = divmod(time.time()-state['time']['start'], 60)
    print(f" |  完成 {state['tokens']['ti']} {state['tokens']['to']} [{int(minutes)}m {int(seconds)}s]")
    return {'paper': {**state['paper'], 'Keywords': keywords}, 'tokens': state['tokens']}


def abstract_node(state: GraphState, config: RunnableConfig):
    """生成摘要和标题"""
    print("正在撰写摘要".ljust(30, '.'), end="", flush=True)
    f_temp1 = Path(f"{state['files']['Temp']}/Abstract.tex")
    f_temp2 = Path(f"{state['files']['Temp']}/Title.tex")

    if f_temp1.exists():
        state['paper']['Abstract'] = temp_file(state, f_temp1, 'read')
        state['paper']['Title']    = temp_file(state, f_temp2, 'read')
        print('已从 Abstract.tex 读取', end="", flush=True)
    else:
        for attempt in range(5):
            print(f'{attempt} ', end="", flush=True)
            PROMPT = abstract_prompt(state, attempt)
            state, result = LLM_call(PROMPT, state)
            try:
                parsed = json_parser3(result)
                state['paper']['Title']    = parsed["Title"]
                state['paper']['Abstract'] = parsed["Abstract"]
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("LLM 多次尝试后仍无法生成有效摘要。")

        # 自我修订
        for _ in range(1):
            PROMPT = abstract_reflection(state)
            state, result = LLM_call(PROMPT, state)
            state['paper']['Abstract'] = extract_latex_block(state, result, "Abstract")
            state['paper']['Abstract'] = fix_percent(state['paper']['Abstract'])

        temp_file(state, f_temp2, 'write', state['paper']['Title'])
        temp_file(state, f_temp1, 'write', state['paper']['Abstract'])
        compile_tex_document(state, f_temp2, state['files']['Temp'])
        success = compile_tex_document(state, f_temp1, state['files']['Temp'])
        if not success:
            state['latex']['section_to_fix'] = 'Abstract'
            state, _ = fix_latex(state, f_temp1)
            state['paper']['Abstract'] = fix_percent(state['paper']['Abstract'])

    save_paper(state, state['files']['Paper_v1'])
    minutes, seconds = divmod(time.time()-state['time']['start'], 60)
    print(f" |  完成 {state['tokens']['ti']} {state['tokens']['to']} [{int(minutes)}m {int(seconds)}s]")
    return {'paper': {**state['paper'], 'Title': state['paper']['Title'],
                      'Abstract': state['paper']['Abstract']},
            'tokens': state['tokens']}


def section_node(state: GraphState, config: RunnableConfig, section_name: str,
                 prompt_fn, reflection_fn=None):
    """通用章节生成节点"""
    print(f'正在撰写{section_name}'.ljust(30, '.'), end="", flush=True)
    f_temp = Path(f"{state['files']['Temp']}/{section_name}.tex")

    if f_temp.exists():
        state['paper'][section_name] = temp_file(state, f_temp, 'read')
        print(f'已从 {section_name}.tex 读取', end="", flush=True)
    else:
        for attempt in range(3):
            print(f'{attempt} ', end="", flush=True)
            PROMPT = prompt_fn(state)
            state, result = LLM_call(PROMPT, state)
            section_text = extract_latex_block(state, result, section_name)
            state['paper'][section_name] = section_text

            if reflection_fn:
                for _ in range(2):
                    PROMPT = reflection_fn(state)
                    state, section_text = LLM_call(PROMPT, state)

            section_text = LaTeX_checker(state, section_text)
            state['paper'][section_name] = clean_section(section_text, section_name)
            temp_file(state, f_temp, 'write', state['paper'][section_name])

            if compile_tex_document(state, f_temp, state['files']['Temp']):
                break
            else:
                state['latex']['section_to_fix'] = section_name
                state, fixed = fix_latex(state, f_temp)
                if fixed:
                    break

    save_paper(state, state['files']['Paper_v1'])
    minutes, seconds = divmod(time.time()-state['time']['start'], 60)
    print(f" |  完成 {state['tokens']['ti']} {state['tokens']['to']} [{int(minutes)}m {int(seconds)}s]")
    return {"paper": {**state["paper"], section_name: state['paper'][section_name]},
            'tokens': state['tokens']}


def introduction_node(state: GraphState, config: RunnableConfig):
    return section_node(state, config, section_name="Introduction",
                        prompt_fn=introduction_prompt,
                        reflection_fn=introduction_reflection)


def methods_node(state: GraphState, config: RunnableConfig):
    return section_node(state, config, section_name="Methods",
                        prompt_fn=methods_prompt, reflection_fn=None)


def results_node(state: GraphState, config: RunnableConfig):
    return section_node(state, config, section_name="Results",
                        prompt_fn=results_prompt, reflection_fn=None)


def conclusions_node(state: GraphState, config: RunnableConfig):
    return section_node(state, config, section_name="Conclusions",
                        prompt_fn=conclusions_prompt, reflection_fn=None)


def image_to_base64(image_path):
    ext = image_path.suffix.lower()
    if ext == '.pdf':
        with fitz.open(str(image_path)) as doc:
            img_bytes = doc.load_page(0).get_pixmap().tobytes("png")
        return base64.b64encode(img_bytes).decode('utf-8')
    elif ext in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    else:
        raise ValueError(f"不支持的图片格式：{image_path}")


def plots_node(state: GraphState, config: RunnableConfig):
    """处理图片：生成图注并插入结果章节"""
    batch_size  = 7
    folder_path = Path(f"{state['files']['Folder']}/{INPUT_FILES}/{state['files']['Plots']}")
    if not folder_path.exists():
        return {'paper': state['paper'], 'tokens': state['tokens']}

    files = [f for f in folder_path.iterdir() if f.is_file() and f.name != '.DS_Store']
    num_images = len(files)
    if num_images > 25:
        random.seed(1)
        files = random.sample(files, 25)
        num_images = 25

    for start in range(0, num_images, batch_size):
        batch_files = files[start:start + batch_size]
        f_temp = Path(f"{state['files']['Temp']}/plots_{start+1}_{min(start+batch_size, num_images)}.json")

        if f_temp.exists():
            images = temp_file(state, f_temp, 'read', json_file=True)
        else:
            images = {}
            for i, file in enumerate(tqdm(batch_files, desc=f"处理图片 {start+1}-{min(start+batch_size, num_images)}")):
                image  = image_to_base64(file)
                PROMPT = caption_prompt(state, image)
                state, result = LLM_call(PROMPT, state)
                caption = extract_latex_block(state, result, "Caption")
                caption = LaTeX_checker(state, caption)
                images[f"image{i}"] = {'name': file.name, 'caption': caption}
            temp_file(state, f_temp, 'write', images, json_file=True)

        print(f'   插入图片 {start+1}-{min(start+batch_size, num_images)}'.ljust(30, '.'), end="", flush=True)
        f_temp = Path(f"{state['files']['Temp']}/Results_{start+1}_{min(start+batch_size, num_images)}.tex")

        if f_temp.exists():
            state['paper']['Results'] = temp_file(state, f_temp, 'read')
        else:
            for attempt in range(3):
                print(f'{attempt} ', end="", flush=True)
                PROMPT = plot_prompt(state, images)
                state, result = LLM_call(PROMPT, state)
                results = extract_latex_block(state, result, "Section")
                results = LaTeX_checker(state, results)
                state['paper']['Results'] = clean_section(results, 'Results')
                if check_images_in_text(state, images):
                    break
            else:
                raise RuntimeError("多次尝试后仍无法将图片插入文本。")
            temp_file(state, f_temp, 'write', state['paper']['Results'])
            save_paper(state, state['files']['Paper_v1'])
            compile_tex_document(state, f_temp, state['files']['Temp'])

        minutes, seconds = divmod(time.time()-state['time']['start'], 60)
        print(f" |  完成 {state['tokens']['ti']} {state['tokens']['to']} [{int(minutes)}m {int(seconds)}s]")

    if num_images > 0:
        print('编译文字+图片'.ljust(30, '.'), end="", flush=True)
        success = compile_tex_document(state, f_temp, state['files']['Temp'])
        if not success:
            state['latex']['section_to_fix'] = "Results"
            state, _ = fix_latex(state, f_temp)
        minutes, seconds = divmod(time.time()-state['time']['start'], 60)
        print(f" |  完成 {state['tokens']['ti']} {state['tokens']['to']} [{int(minutes)}m {int(seconds)}s]")

    compile_latex(state, state['files']['Paper_v1'])
    return {'paper': {**state['paper'], 'Results': state['paper']['Results']},
            'tokens': state['tokens']}


def refine_results(state: GraphState, config: RunnableConfig):
    """完善结果章节（整合文字与图片）"""
    if state['files']['num_plots'] == 0:
        save_paper(state, state['files']['Paper_v2'])
        compile_latex(state, state['files']['Paper_v2'])
        return state

    print('正在完善结果章节'.ljust(30, '.'), end="", flush=True)
    f_temp = Path(f"{state['files']['Temp']}/Results_refined.tex")

    if f_temp.exists():
        state['paper']['Results'] = temp_file(state, f_temp, 'read')
    else:
        for attempt in range(3):
            print(f'{attempt} ', end="", flush=True)
            PROMPT = refine_results_prompt(state)
            state, result = LLM_call(PROMPT, state)
            results = extract_latex_block(state, result, "Results")
            results = LaTeX_checker(state, results)
            section_text = clean_section(results, 'Results')
            state['paper']['Results'] = _check_references(state, section_text)
            temp_file(state, f_temp, 'write', state['paper']['Results'])
            if compile_tex_document(state, f_temp, state['files']['Temp']):
                break
            else:
                state['latex']['section_to_fix'] = "Results"
                state, fixed = fix_latex(state, f_temp)
                if fixed:
                    break

    minutes, seconds = divmod(time.time()-state['time']['start'], 60)
    print(f" |  完成 {state['tokens']['ti']} {state['tokens']['to']} [{int(minutes)}m {int(seconds)}s]")
    save_paper(state, state['files']['Paper_v2'])
    compile_latex(state, state['files']['Paper_v2'])
    return {'paper': {**state['paper'], 'Results': state['paper']['Results']},
            'tokens': state['tokens']}


def _check_references(state: GraphState, text: str) -> str:
    PROMPT = references_prompt(state, text)
    state, result = LLM_call(PROMPT, state)
    return extract_latex_block(state, result, "Text")


async def _add_citations_async(state, text, section_name):
    f_temp1 = Path(f"{state['files']['Temp']}/{section_name}_w_citations.tex")
    f_temp2 = Path(f"{state['files']['Temp']}/{section_name}.bib")
    if f_temp1.exists():
        new_text   = temp_file(state, f_temp1, 'read')
        references = temp_file(state, f_temp2, 'read')
    else:
        if not _has_citations:
            return section_name, text, ""
        loop = asyncio.get_event_loop()
        func = partial(process_tex_file_with_references, text, state["keys"])
        new_text, references = await loop.run_in_executor(None, func)
        new_text = clean_section(new_text, section_name)
        temp_file(state, f_temp2, 'write', references)
        temp_file(state, f_temp1, 'write', new_text)
    print(f'    {section_name} 完成')
    return section_name, new_text, references


async def citations_node(state: GraphState, config: RunnableConfig):
    """异步添加引用文献"""
    print("正在添加参考文献...")
    sections = ['Introduction', 'Methods']
    tasks   = [_add_citations_async(state, state['paper'][s], s) for s in sections]
    results = await asyncio.gather(*tasks)

    bib_set  = set()
    bib_list = []
    for section_name, updated_text, references in results:
        state['paper'][section_name] = updated_text
        for entry in references.strip().split('\n\n'):
            clean = entry.strip()
            if clean and clean not in bib_set:
                bib_set.add(clean)
                bib_list.append(clean)
    state['paper']['References'] = "\n\n".join(bib_list)

    save_paper(state, state['files']['Paper_v3'])
    save_bib(state)
    process_bib_file(f"{state['files']['Paper_folder']}/bibliography_temp.bib",
                     f"{state['files']['Paper_folder']}/bibliography.bib")
    print("✅ 已成功添加参考文献。")
    compile_latex(state, state['files']['Paper_v3'])

    print("最终章节检查...")
    for section_name in sections:
        f_temp = Path(f"{state['files']['Temp']}/{section_name}_w_citations2.tex")
        if f_temp.exists():
            section_text = temp_file(state, f_temp, 'read')
        else:
            PROMPT = clean_section_prompt(state, state['paper'][section_name])
            state, result = LLM_call(PROMPT, state)
            section_text = extract_latex_block(state, result, "Text")
            section_text = LaTeX_checker(state, section_text)
            section_text = clean_section(section_text, section_name)
            temp_file(state, f_temp, 'write', section_text)
        state['paper'][section_name] = section_text

    save_paper(state, state['files']['Paper_v4'])
    compile_latex(state, state['files']['Paper_v4'])
    return {'paper': state['paper'], 'tokens': state['tokens']}
