"""系统设置页：API 密钥、项目管理。"""
import os
import shutil
import streamlit as st
from utils import extract_api_keys, set_api_keys
from constants import LLMs, PROJECTS_ROOT

yz = st.session_state.yz
current = getattr(st.session_state, "current_project", "")

st.title("⚙️ 系统设置")
st.divider()

# ---------------------------------------------------------------------------
# API 密钥
# ---------------------------------------------------------------------------
st.subheader("🔑 API 密钥")
st.markdown("密钥仅在当前会话内有效，不会持久化存储。")

llm_labels = {
    "GEMINI":     "Google Gemini",
    "OPENAI":     "OpenAI",
    "ANTHROPIC":  "Anthropic",
    "PERPLEXITY": "Perplexity（用于参考文献）",
    "MINIMAX":    "MiniMax",
}
cols = st.columns(2)
for i, llm in enumerate(LLMs + ["MINIMAX"]):
    with cols[i % 2]:
        key = st.text_input(
            f"{llm_labels.get(llm, llm)}",
            type="password",
            key=f"settings_{llm}_key",
            value=getattr(yz.keys, llm, "") or "",
        )
        if key:
            set_api_keys(yz.keys, key, llm)

st.divider()
st.subheader("🤗 HuggingFace")
st.markdown("下载受限（gated）数据集时需要填入 HF Token。[获取 Token](https://huggingface.co/settings/tokens)")
hf_token = st.text_input(
    "HuggingFace Token（hf_...）",
    type="password",
    key="settings_HF_TOKEN",
    value=getattr(yz.keys, "HF_TOKEN", "") or "",
)
if hf_token:
    set_api_keys(yz.keys, hf_token, "HF_TOKEN")

# .env 文件批量导入
with st.expander("📄 上传 .env 文件批量导入"):
    st.markdown("""格式：
```
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="..."
GEMINI_API_KEY="..."
PERPLEXITY_API_KEY="..."
```""")
    uploaded_env = st.file_uploader(".env 文件", accept_multiple_files=False, key="upload_env")
    if uploaded_env:
        keys = extract_api_keys(uploaded_env)
        for k, v in keys.items():
            set_api_keys(yz.keys, v, k)
        st.success(f"✅ 已导入 {len(keys)} 个密钥：{list(keys.keys())}")

st.divider()

# ---------------------------------------------------------------------------
# 上传数据文件
# ---------------------------------------------------------------------------
st.subheader("📂 上传数据文件")
uploaded_data = st.file_uploader(
    "上传数据文件（CSV、JSON、Excel 等）",
    accept_multiple_files=True,
    key="settings_data_upload",
)
if uploaded_data:
    data_dir = os.path.join(yz.project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for f in uploaded_data:
        with open(os.path.join(data_dir, f.name), "wb") as fp:
            fp.write(f.getbuffer())
    st.success(f"✅ 已上传 {len(uploaded_data)} 个文件")

st.divider()

# ---------------------------------------------------------------------------
# 项目管理
# ---------------------------------------------------------------------------
st.subheader("📁 项目管理")

if os.path.exists(PROJECTS_ROOT):
    projects = sorted([d for d in os.listdir(PROJECTS_ROOT)
                       if os.path.isdir(os.path.join(PROJECTS_ROOT, d))])

    for proj in projects:
        proj_dir = os.path.join(PROJECTS_ROOT, proj)
        is_current = proj == current
        label = f"**{proj}**" + (" ← 当前项目" if is_current else "")
        with st.expander(label):
            # 文件统计
            file_count = sum(1 for _, _, files in os.walk(proj_dir) for _ in files)
            st.caption(f"共 {file_count} 个文件")

            # 历史版本统计
            history_dir = os.path.join(proj_dir, "history")
            if os.path.exists(history_dir):
                hist_count = len(os.listdir(history_dir))
                st.caption(f"历史备份：{hist_count} 个版本")

            if not is_current:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("切换到此项目", key=f"switch_{proj}", use_container_width=True):
                        st.session_state.current_project = proj
                        if "yz" in st.session_state:
                            del st.session_state.yz
                        st.rerun()
                with col2:
                    if st.button("🗑️ 删除", key=f"delete_{proj}", use_container_width=True,
                                 type="secondary"):
                        st.session_state[f"confirm_delete_{proj}"] = True

                if st.session_state.get(f"confirm_delete_{proj}"):
                    st.warning(f"⚠️ 确认删除项目「{proj}」？此操作不可恢复！")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("确认删除", key=f"confirm_del_{proj}", type="primary"):
                            shutil.rmtree(proj_dir, ignore_errors=True)
                            del st.session_state[f"confirm_delete_{proj}"]
                            st.success(f"✅ 已删除「{proj}」")
                            st.rerun()
                    with col_b:
                        if st.button("取消", key=f"cancel_del_{proj}"):
                            del st.session_state[f"confirm_delete_{proj}"]
                            st.rerun()
            else:
                st.info("这是当前活动项目，无法删除。")

st.divider()

# 项目状态
st.subheader("📋 当前项目状态")
files_status = {
    "研究方向": "input_files/data_description.md",
    "文献报告": "input_files/literature.md",
    "研究想法": "input_files/idea.md",
    "研究方法": "input_files/methods.md",
    "实验结果": "input_files/results.md",
    "论文 PDF":  "paper/paper_v4_final.pdf",
    "审稿报告": "input_files/referee.md",
}
cols = st.columns(4)
for i, (label, rel_path) in enumerate(files_status.items()):
    with cols[i % 4]:
        full_path = os.path.join(yz.project_dir, rel_path)
        if os.path.exists(full_path):
            st.success(f"✅ {label}")
        else:
            st.info(f"⬜ {label}")
