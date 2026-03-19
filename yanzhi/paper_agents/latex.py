import subprocess
import os
import re
from pathlib import Path

from .parameters import GraphState
from .prompts import fix_latex_bug_prompt
from .tools import LLM_call, extract_latex_block, temp_file
from .journal import LatexPresets
from .latex_presets import journal_dict

special_chars = {
    "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
    "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\~{}", "^": r"\^{}",
}


def extract_latex_errors(state):
    with open(state['files']['LaTeX_log'], 'r') as f:
        lines = f.readlines()
    errors = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("! "):
            error_block = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if (next_line.startswith("! ") or next_line.startswith("(/") or
                        next_line.startswith(")") or next_line.startswith("Package ") or
                        next_line.startswith("Document Class") or
                        re.match(r'^\([^\)]+\.sty', next_line) or
                        re.match(r'^\(/', next_line) or
                        re.match(r'^.*\.tex$', next_line)):
                    break
                error_block.append(next_line)
                i += 1
            errors.append("\n".join(error_block))
        else:
            i += 1
    with open(state['files']['LaTeX_err'], 'w') as f:
        if errors:
            f.write("LaTeX 编译错误：\n\n")
            for error in errors:
                f.write(error + "\n\n")
        else:
            f.write("✅ 未发现 LaTeX 错误。\n")


def clean_files(doc_name, doc_folder):
    file_path = Path(doc_name)
    doc_stem = file_path.stem
    for suffix in ['aux', 'log', 'pdf', 'out']:
        fp = f'{doc_folder}/{doc_stem}.{suffix}'
        if os.path.exists(fp):
            os.system(f'rm {fp}')


def compile_tex_document(state: dict, doc_name: str, doc_folder: str) -> None:
    file_path = Path(doc_name)
    doc_name  = file_path.name
    doc_stem  = file_path.stem
    bib_path  = os.path.join(state['files']['Temp'], "bibliography.bib")

    def run_xelatex():
        result = subprocess.run(["xelatex", doc_name], cwd=doc_folder,
                                input="\n", capture_output=True, text=True)
        if result.returncode != 0:
            print("❌", end="", flush=True)
            clean_files(doc_name, doc_folder)
            log_output(result)
            extract_latex_errors(state)
            return False
        return True

    def run_bibtex():
        result = subprocess.run(["bibtex", doc_stem], cwd=doc_folder,
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"BibTeX 失败：{result.stderr}")

    def log_output(result):
        with open(state['files']['LaTeX_log'], 'a') as f:
            f.write("---- STDOUT ----\n")
            f.write(result.stdout)
            f.write("---- STDERR ----\n")
            f.write(result.stderr)

    if not run_xelatex():
        return False
    if os.path.exists(bib_path):
        run_bibtex()
        total_passes = 3
    else:
        total_passes = 2
    for _ in range(2, total_passes + 1):
        run_xelatex()
    print("✅", end="", flush=True)
    clean_files(doc_name, doc_folder)
    return True


def compile_latex(state: GraphState, paper_name: str) -> None:
    paper_stem = Path(paper_name).stem

    def run_xelatex():
        return subprocess.run(["xelatex", "-interaction=nonstopmode", "-file-line-error", paper_name],
                              cwd=state['files']['Paper_folder'],
                              input="\n", capture_output=True, text=True, check=True)

    def run_bibtex():
        subprocess.run(["bibtex", paper_stem], cwd=state['files']['Paper_folder'],
                       capture_output=True, text=True)

    def log_output(i, result_or_error, is_error=False):
        with open(state['files']['LaTeX_log'], 'a') as f:
            f.write(f"\n==== {'错误' if is_error else '通过'} 第 {i} 次 ====\n")
            f.write("---- STDOUT ----\n")
            f.write(result_or_error.stdout or "")
            f.write("---- STDERR ----\n")
            f.write(result_or_error.stderr or "")

    print(f'编译 {paper_stem}'.ljust(33, '.'), end="", flush=True)
    try:
        run_xelatex()
        print("✅", end="", flush=True)
    except subprocess.CalledProcessError as e:
        log_output("第1次", e, is_error=True)
        print("❌", end="", flush=True)

    further_iterations = 1
    if os.path.exists(f"{state['files']['Paper_folder']}/bibliography.bib"):
        run_bibtex()
        further_iterations = 2

    for i in range(further_iterations):
        try:
            run_xelatex()
            print("✅", end="", flush=True)
        except subprocess.CalledProcessError as e:
            log_output(f"第{i+2}次", e, is_error=True)
            print("❌", end="", flush=True)

    for fin in [f'{paper_stem}.aux', f'{paper_stem}.log', f'{paper_stem}.out',
                f'{paper_stem}.bbl', f'{paper_stem}.blg', f'{paper_stem}.synctex.gz']:
        fp = f"{state['files']['Paper_folder']}/{fin}"
        if os.path.exists(fp):
            os.remove(fp)
    print("")


def save_paper(state: GraphState, paper_name: str):
    """组装完整 LaTeX 论文并保存（中文章节标题）"""
    journaldict: LatexPresets = journal_dict[state['paper']['journal']]

    author      = "中文科研助手"
    affiliation = "人工智能研究院"

    paper = rf"""\documentclass[{journaldict.layout}]{{{journaldict.article}}}

\usepackage{{amsmath}}
\usepackage{{multirow}}
\usepackage{{natbib}}
\usepackage{{graphicx}}
\usepackage{{tabularx}}
{journaldict.usepackage}


\begin{{document}}

{journaldict.title}{{{state['paper'].get('Title','')}}}

{journaldict.author(author)}
{journaldict.affiliation(affiliation)}

{journaldict.abstract(state['paper'].get('Abstract',''))}
{journaldict.keywords(state['paper']['Keywords'])}


\section{{引言}}
\label{{sec:intro}}
{state['paper'].get('Introduction','')}

\section{{研究方法}}
\label{{sec:methods}}
{state['paper'].get('Methods','')}

\section{{结果与讨论}}
\label{{sec:results}}
{state['paper'].get('Results','')}

\section{{结论}}
\label{{sec:conclusions}}
{state['paper'].get('Conclusions','')}

\bibliography{{bibliography}}{{}}
{journaldict.bibliographystyle}

\end{{document}}
"""
    f_in = f"{state['files']['Paper_folder']}/{paper_name}"
    with open(f_in, 'w', encoding='utf-8') as f:
        f.write(paper)


def save_bib(state: GraphState):
    with open(f"{state['files']['Paper_folder']}/bibliography_temp.bib", 'a', encoding='utf-8') as f:
        f.write(state['paper']['References'].strip() + "\n")


def escape_special_chars(text):
    parts = re.split(r'(\$.*?\$)', text)
    sanitized = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            sanitized.append(part)
        else:
            for char, escaped in special_chars.items():
                part = part.replace(char, escaped)
            sanitized.append(part)
    return ''.join(sanitized)


def process_bib_file(input_file, output_file):
    with open(input_file, 'r') as fin:
        lines = fin.readlines()
    processed_lines = []
    for line in lines:
        if line.strip().startswith('title') or line.strip().startswith('journal'):
            key, value = line.split('=', 1)
            content = re.search(r'[{\"](.+)[}\"]', value).group(1)
            escaped_content = escape_special_chars(content)
            escaped_content = re.sub(r'\b([A-Z]{2,})\b', r'{\1}', escaped_content)
            processed_lines.append(f'  {key.strip()} = {{{escaped_content}}},\n')
        else:
            processed_lines.append(line)
    with open(output_file, 'w') as fout:
        fout.writelines(processed_lines)


def fix_latex(state, f_temp):
    file_path = Path(f_temp)
    f_stem    = file_path.with_suffix('')
    suffix    = file_path.suffix
    for i in range(3):
        PROMPT = fix_latex_bug_prompt(state)
        state, result = LLM_call(PROMPT, state)
        fixed_text = extract_latex_block(state, result, "Text")
        state['paper'][state['latex']['section_to_fix']] = fixed_text
        f_name = f"{f_stem}_v{i+1}{suffix}"
        temp_file(state, f_name, 'write', fixed_text)
        if compile_tex_document(state, f_name, state['files']['Temp']):
            os.system(f'mv {f_temp} {f_stem}_orig{suffix}')
            os.system(f"mv {f_name} {f_temp}")
            return state, True
    return state, False


def fix_percent(text):
    return re.sub(r'(?<!\\)%', r'\\%', text)
