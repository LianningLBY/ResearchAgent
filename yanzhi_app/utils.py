import os
import io
import zipfile
import re
import sys
import uuid
import shutil
import time
from contextlib import contextmanager
import streamlit as st
from yanzhi import KeyManager

from constants import PROJECT_DIR, LLMs


def show_markdown_file(file_path: str, extra_format: bool = False, label: str = "") -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        response = f.read()
    if extra_format:
        response = response.replace("\n研究想法：\n\t", "### 研究想法\n").replace("\t\t", "    ")
    st.download_button(
        label="下载 " + label,
        data=response,
        file_name=os.path.basename(file_path),
        mime="text/plain",
        icon=":material/download:",
    )
    st.markdown(response)


def extract_api_keys(uploaded_file):
    pattern = re.compile(r'^\s*([A-Z_]+_API_KEY)\s*=\s*"([^"]+)"')
    keys = {}
    content = uploaded_file.read().decode("utf-8").split("\n")
    for line in content:
        match = pattern.match(line)
        if match:
            key_name, key_value = match.groups()
            key_name = key_name.replace("_API_KEY", "")
            if key_name in LLMs:
                keys[key_name] = key_value
            if "GOOGLE" in key_name:
                keys["GEMINI"] = key_value
    return keys


def set_api_keys(key_manager: KeyManager, api_key: str, llm: str):
    if llm == "GEMINI":
        key_manager.GEMINI = api_key
    elif llm == "OPENAI":
        key_manager.OPENAI = api_key
    elif llm == "ANTHROPIC":
        key_manager.ANTHROPIC = api_key
    elif llm == "PERPLEXITY":
        key_manager.PERPLEXITY = api_key


def create_zip_in_memory(folder_path: str):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer


def get_project_dir():
    if "project_dir" not in st.session_state:
        temp_dir = f"project_dir_{uuid.uuid4().hex}"
        os.makedirs(temp_dir, exist_ok=True)
        st.session_state.project_dir = temp_dir
    return st.session_state.project_dir


class StreamToBuffer(io.StringIO):
    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback

    def write(self, s):
        super().write(s)
        self.seek(0)
        self.update_callback(self.read())
        self.seek(0, io.SEEK_END)


def _escape_log(text: str) -> str:
    """HTML 转义日志文本，防止 LaTeX/Markdown 符号乱渲染。"""
    import html
    return html.escape(text).replace("\n", "<br>")


@contextmanager
def stream_to_streamlit(container):
    buffer = StreamToBuffer(
        update_callback=lambda text: container.markdown(
            f'<div class="log-box">{_escape_log(text)}</div>',
            unsafe_allow_html=True,
        )
    )
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        yield
    finally:
        sys.stdout = old_stdout


def get_latest_mtime_in_folder(folder_path: str) -> float:
    latest_mtime = os.path.getmtime(folder_path)
    for root, dirs, files in os.walk(folder_path):
        for name in files + dirs:
            full_path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(full_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
            except FileNotFoundError:
                continue
    return latest_mtime


def show_history(yz, file_type: str, on_load=None):
    """
    展示某类文件的历史版本，并提供加载按钮。

    Args:
        yz: YanZhi 实例
        file_type: 文件类型前缀，如 'idea'、'methods'、'literature'
        on_load: 加载历史版本后的回调，接收文件路径
    """
    history = yz.list_history(file_type)
    if not history:
        return

    with st.expander(f"📜 历史版本（共 {len(history)} 条）"):
        for i, item in enumerate(history[:15]):
            ts = item["timestamp"]
            try:
                from datetime import datetime
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                label = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                label = ts

            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"🕐 {label}")
                preview = item["preview"].strip()
                if preview:
                    st.markdown(
                        f"<small style='color:#6b7280'>{preview[:120].replace(chr(10),' ')}…</small>",
                        unsafe_allow_html=True,
                    )
            with col2:
                if st.button("加载", key=f"hist_{file_type}_{i}", use_container_width=True):
                    try:
                        with open(item["path"], "r", encoding="utf-8") as f:
                            content = f.read()
                        if on_load:
                            on_load(content)
                        st.success("✅ 历史版本已加载")
                        st.rerun()
                    except Exception as e:
                        st.error(f"加载失败：{e}")
            st.divider()


def chat_section(yz, page_key: str, content_type: str, get_current_content, on_save, llm_model: str):
    """
    渲染对话式修改界面。

    Args:
        yz: YanZhi 实例
        page_key: 页面唯一标识（如 'idea'、'methods'），用于隔离 session_state
        content_type: 内容类型显示名（如 '研究想法'、'研究方法'）
        get_current_content: callable，返回当前最新内容（优先用最后一条 AI 回复）
        on_save: callable(content)，保存内容到文件
        llm_model: 模型名称字符串
    """
    chat_key  = f"chat_{page_key}"
    draft_key = f"chat_{page_key}_draft"   # 最新 AI 草稿（待确认）

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.subheader("💬 对话修改")
    st.caption("直接告诉 AI 你想怎么改，AI 会输出修改后的完整内容，确认后点「保存此版本」。")

    # ── 显示对话历史 ──────────────────────────────────────────────────
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"],
                             avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            if msg.get("is_revision"):
                # AI 给出的完整修订内容用代码块展示，带保存按钮
                st.markdown(msg["content"])
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.caption(f"↑ 以上是修改后的完整{content_type}")
                with col2:
                    if st.button("✅ 保存此版本", key=f"save_{page_key}_{msg['idx']}",
                                 type="primary", use_container_width=True):
                        on_save(msg["content"])
                        st.success(f"✅ 已保存！历史版本已自动备份。")
                        st.rerun()
            else:
                st.markdown(msg["content"])

    # ── 清空对话按钮 ──────────────────────────────────────────────────
    if st.session_state[chat_key]:
        if st.button("🗑️ 清空对话记录", key=f"clear_chat_{page_key}"):
            st.session_state[chat_key] = []
            st.rerun()

    # ── 对话输入框 ────────────────────────────────────────────────────
    user_input = st.chat_input(
        f"告诉我怎么修改这份{content_type}…",
        key=f"chat_input_{page_key}",
    )

    if user_input:
        current = get_current_content()
        if not current or not current.strip():
            st.warning(f"请先生成{content_type}，再进行对话修改。")
            return

        # 追加用户消息
        st.session_state[chat_key].append({
            "role": "user",
            "content": user_input,
            "idx": len(st.session_state[chat_key]),
        })

        # 构建历史（不含 is_revision 标记，只传 role+content）
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state[chat_key][:-1]
        ]

        with st.spinner(f"正在修改{content_type}…"):
            try:
                revised = yz.chat_revise(
                    content_type=content_type,
                    current_content=current,
                    user_message=user_input,
                    chat_history=history if history else None,
                    llm=llm_model,
                )
                idx = len(st.session_state[chat_key])
                st.session_state[chat_key].append({
                    "role": "assistant",
                    "content": revised,
                    "is_revision": True,
                    "idx": idx,
                })
            except Exception as e:
                st.error(f"修改失败：{e}")

        st.rerun()


def delete_old_folders(days_old: int = 1):
    now = time.time()
    cutoff = now - (days_old * 86400)
    for entry in os.listdir("."):
        if os.path.isdir(entry) and entry.startswith("project_dir_"):
            latest_mtime = get_latest_mtime_in_folder(entry)
            if latest_mtime < cutoff:
                shutil.rmtree(entry)
