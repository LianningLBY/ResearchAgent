import re
import sys
import json
import time
import json5
from pathlib import Path

from .prompts import fixer_prompt, LaTeX_prompt
from .parameters import GraphState
from .journal import LatexPresets
from .latex_presets import journal_dict
from ..log import get_logger

logger = get_logger(__name__)


def _write_token_record(state: dict, input_tok: int, output_tok: int) -> None:
    """将本次 LLM 调用的 token 用量以 JSON Lines 格式追加到记录文件。"""
    record = {
        "ts":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": state["llm"].get("model", "unknown"),
        "i":     input_tok,
        "o":     output_tok,
        "ti":    state["tokens"]["ti"],
        "to":    state["tokens"]["to"],
    }
    with open(state["files"]["LLM_calls"], "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def LLM_call(prompt, state):
    last_err = None
    for attempt in range(3):
        try:
            message = state['llm']['llm'].invoke(prompt)
            input_tokens  = message.usage_metadata.get('input_tokens', 0)
            output_tokens = message.usage_metadata.get('output_tokens', 0)
            if output_tokens >= state['llm']['max_output_tokens']:
                logger.warning("已达到最大输出 token 数（%d）！输出可能被截断。", state['llm']['max_output_tokens'])
            state['tokens']['ti'] += input_tokens
            state['tokens']['to'] += output_tokens
            state['tokens']['i']   = input_tokens
            state['tokens']['o']   = output_tokens
            _write_token_record(state, input_tokens, output_tokens)
            return state, message.content
        except Exception as e:
            last_err = e
            logger.warning("LLM_call 失败（%d/3）：%s，3秒后重试...", attempt + 1, e)
            time.sleep(3)
    logger.error("LLM_call 三次均失败：%s", last_err)
    return state, ""


def LLM_call_stream(prompt, state):
    output_file_path = state['files']['f_stream']
    full_content = ''
    state['tokens']['i'] = 0
    state['tokens']['o'] = 0
    try:
        with open(output_file_path, 'a', encoding='utf-8') as f:
            for chunk in state['llm']['llm'].stream(prompt):
                # chunk.content may be a list (Anthropic) or string (Gemini/OpenAI)
                raw = chunk.content
                if isinstance(raw, list):
                    text = ''.join(
                        item.get('text', '') if isinstance(item, dict) else str(item)
                        for item in raw
                    )
                elif isinstance(raw, str):
                    text = raw
                else:
                    text = str(raw) if raw else ''
                if text:
                    f.write(text)
                    f.flush()
                    if state['llm'].get('stream_verbose', False):
                        print(text, end='', flush=True)
                    full_content += text
                usage = chunk.usage_metadata if hasattr(chunk, 'usage_metadata') else None
                if usage:
                    it = usage.get('input_tokens', 0)
                    ot = usage.get('output_tokens', 0)
                    if ot >= state['llm']['max_output_tokens']:
                        logger.warning("已达到最大输出 token 数（%d）！输出可能被截断。", state['llm']['max_output_tokens'])
                    state['tokens']['ti'] += it
                    state['tokens']['to'] += ot
                    state['tokens']['i']  += it
                    state['tokens']['o']  += ot
            f.write('\n\n')
    except Exception as stream_err:
        logger.warning("流式调用异常，回退到非流式调用：%s", stream_err)
        full_content = ''

    # 流式返回为空时，回退到非流式调用
    if not full_content.strip():
        import sys as _sys
        print("⚠️  [tools] 流式调用未返回内容，尝试非流式调用…", file=_sys.stderr, flush=True)
        logger.warning("流式调用未返回内容，尝试非流式调用…")
        try:
            message = state['llm']['llm'].invoke(prompt)
            raw = message.content
            if isinstance(raw, list):
                full_content = ''.join(
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in raw
                )
            else:
                full_content = str(raw) if raw else ''
            usage = message.usage_metadata if hasattr(message, 'usage_metadata') else {}
            it = usage.get('input_tokens', 0) if usage else 0
            ot = usage.get('output_tokens', 0) if usage else 0
            state['tokens']['ti'] += it
            state['tokens']['to'] += ot
            state['tokens']['i']   = it
            state['tokens']['o']   = ot
            with open(output_file_path, 'a', encoding='utf-8') as f:
                f.write(full_content + '\n\n')
            if state['llm'].get('stream_verbose', False):
                print(full_content, flush=True)
            logger.info("非流式调用成功，返回 %d 字符。", len(full_content))
        except Exception as invoke_err:
            import sys as _sys2
            print(f"❌ [tools] 非流式调用失败：{invoke_err}", file=_sys2.stderr, flush=True)
            logger.error("非流式调用也失败：%s", invoke_err)

    _write_token_record(state, state['tokens']['i'], state['tokens']['o'])
    return state, full_content


def temp_file(state, fin, action, text=None, json_file=False):
    journaldict: LatexPresets = journal_dict[state['paper']['journal']]
    if action == 'read':
        with open(fin, 'r', encoding='utf-8') as f:
            if json_file:
                return json.load(f)
            latex_text = f.read()
            match = re.search(r'\\begin{document}(.*?)\\end{document}', latex_text, re.DOTALL)
            if match:
                return match.group(1).strip()
            raise ValueError("在文件中未找到 LaTeX document 正文块！")
    elif action == 'write':
        with open(fin, 'w', encoding='utf-8') as f:
            if json_file:
                json.dump(text, f, indent=2, ensure_ascii=False)
            else:
                latex_text = rf"""\documentclass[{journaldict.layout}]{{{journaldict.article}}}

\usepackage{{amsmath}}
\usepackage{{multirow}}
\usepackage{{natbib}}
\usepackage{{graphicx}}
{journaldict.usepackage}

\begin{{document}}

{text}

\end{{document}}
"""
                f.write(latex_text)
    else:
        raise ValueError(f"action 参数错误：'{action}'，仅支持 'read' 或 'write'。")


def json_parser(text):
    json_pattern = r"```json(.*)```"
    match = re.findall(json_pattern, text, re.DOTALL)
    json_string = match[0].strip().replace("\\", "\\\\")
    try:
        return json.loads(json_string)
    except json.decoder.JSONDecodeError:
        try:
            return json.loads(json_string.replace("'", "\""))
        except Exception as e:
            raise ValueError(f"JSON 解析失败：{e}")


def json_parser2(text: str):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        raise ValueError("未找到 JSON 代码块。")
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        snippet = m.group(1)[max(0, e.pos-40):e.pos+40]
        raise ValueError(f"JSON 解析错误 pos {e.pos}: {e.msg}\n…{snippet}…")


def _extract_json_object(text: str) -> str | None:
    """通过括号计数精确提取第一个完整 JSON 对象，不受非贪婪正则截断影响。"""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def json_parser3(text: str):
    # 1. 直接解析（LLM 返回纯 JSON）
    try:
        return json5.loads(text.strip())
    except Exception:
        pass

    # 2. 提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json5.loads(m.group(1))
        except Exception:
            pass

    # 3. 括号计数提取（最健壮，处理代码块外的 JSON 以及 LLM 额外文字）
    raw = _extract_json_object(text)
    if raw:
        return json5.loads(raw)

    raise ValueError("未找到 JSON 代码块。")


def extract_latex_block(state: GraphState, text: str, block: str) -> str:
    if isinstance(text, list):
        text = "".join([str(item) for item in text])
    pattern = rf"\\begin{{{block}}}(.*?)\\end{{{block}}}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    with open(state['files']['Error'], 'w', encoding='utf-8') as f:
        f.write(text)
    try:
        return fixer(state, block)
    except ValueError:
        raise ValueError(f"无法提取 {block} 块")


def fixer(state: GraphState, section_name):
    path = Path(state['files']['Error'])
    with path.open("r", encoding="utf-8") as f:
        Text = f.read()
    PROMPT = fixer_prompt(Text, section_name)
    state, result = LLM_call(PROMPT, state)
    pattern = rf"\\begin{{{section_name}}}(.*?)\\end{{{section_name}}}"
    match = re.search(pattern, result, re.DOTALL)
    if match:
        return match.group(1).strip()
    with open(state['files']['Error'], 'w', encoding='utf-8') as f:
        f.write(result)
    logger.error("LaTeX 修复器未能提取 %s 块，终止执行。", section_name)
    sys.exit(1)


def LaTeX_checker(state, text):
    PROMPT = LaTeX_prompt(text)
    state, result = LLM_call(PROMPT, state)
    text = extract_latex_block(state, result, "Text")
    return text


def clean_section(text, section):
    for s in [r"\documentclass{article}", r"\begin{document}", r"\end{document}",
              r"\maketitle", r"<PARAGRAPH>", r"</PARAGRAPH>",
              r"```latex", r"```", r"\usepackage{amsmath}"]:
        text = text.replace(s, "")
    for tmpl in [fr"\section{{{section}}}", fr"\section*{{{section}}}",
                 fr"\begin{{{section}}}", fr"\end{{{section}}}",
                 fr"</{section}>", fr"<{section}>"]:
        text = text.replace(tmpl, "")
    return text


def check_images_in_text(state, images):
    for key, value in images.items():
        if value["name"] not in state['paper']['Results']:
            return False
    return True
