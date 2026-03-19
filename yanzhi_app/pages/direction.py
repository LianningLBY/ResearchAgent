"""研究方向输入页：支持直接输入、文件上传、以及对话辅助填写。"""
import os
import streamlit as st
from yanzhi import models
from utils import show_markdown_file, chat_section

yz = st.session_state.yz
desc_path = os.path.join(yz.project_dir, "input_files", "data_description.md")

st.title("🎯 研究方向")

tab_input, tab_chat = st.tabs(["✍️ 输入", "💬 对话辅助"])

# ---------------------------------------------------------------------------
# 输入标签页
# ---------------------------------------------------------------------------
with tab_input:
    st.markdown("设定本次研究的出发点。可以是具体数据集描述，也可以是一个模糊的研究方向。")

    mode = st.radio(
        "研究起点",
        ["🧭 从研究方向出发（无数据）", "📂 从已有数据出发"],
        horizontal=True,
        key="direction_mode",
    )

    current_description = ""
    try:
        with open(desc_path, "r", encoding="utf-8") as f:
            current_description = f.read()
    except FileNotFoundError:
        pass

    if mode == "🧭 从研究方向出发（无数据）":
        st.info("💡 填写研究方向后，建议先前往「文献调研」了解领域现状，再生成研究想法。")
        data_descr = st.text_area(
            "研究方向或科学问题",
            placeholder=(
                "例如：\n"
                "探索 Transformer 架构在小样本医学图像分类中的应用，"
                "重点关注域适应方法的改进。\n\n"
                "或者：\n"
                "研究大语言模型在中文法律文书理解中的局限性，"
                "分析其在复杂推理场景下的错误模式。"
            ),
            value=current_description,
            key="data_descr_direction",
            height=180,
        )
    else:
        data_descr = st.text_area(
            "数据集和工具描述",
            placeholder=(
                "例如：\n"
                "使用存储在 /data/climate.csv 的气候观测数据（2000-2020年），"
                "包含温度、降水、风速等变量。\n"
                "工具：Python、pandas、scikit-learn。"
            ),
            value=current_description,
            key="data_descr_data",
            height=180,
        )
        uploaded_file = st.file_uploader("或上传 Markdown 格式描述文件", key="upload_description")
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            yz.set_data_description(content)
            st.success("✅ 描述文件已上传！")
            st.rerun()

        with st.expander("📎 上传数据文件（可选）"):
            uploaded_data = st.file_uploader("上传数据文件", accept_multiple_files=True,
                                             key="upload_data_files")
            if uploaded_data:
                data_dir = os.path.join(yz.project_dir, "data")
                os.makedirs(data_dir, exist_ok=True)
                for f in uploaded_data:
                    with open(os.path.join(data_dir, f.name), "wb") as fp:
                        fp.write(f.getbuffer())
                st.success(f"✅ 已上传 {len(uploaded_data)} 个文件")

    if st.button("💾 保存研究方向", type="primary", key="save_direction"):
        if data_descr and data_descr.strip():
            yz.set_data_description(data_descr)
            st.success("✅ 研究方向已保存！")
            if mode == "🧭 从研究方向出发（无数据）":
                st.info("👉 下一步：前往「文献调研」搜索相关文献")
            else:
                st.info("👉 下一步：前往「研究想法」生成研究想法")
        else:
            st.warning("请先填写内容。")

    st.divider()
    st.subheader("已保存的研究方向")
    try:
        show_markdown_file(desc_path, label="研究方向")
    except FileNotFoundError:
        st.caption("尚未设置研究方向。")

# ---------------------------------------------------------------------------
# 对话辅助标签页：用 AI 帮助写研究方向
# ---------------------------------------------------------------------------
with tab_chat:
    st.markdown("不确定怎么描述？和 AI 对话，让它帮你梳理和完善研究方向。")

    model_keys = list(models.keys())
    chat_llm = st.selectbox(
        "对话模型",
        model_keys,
        index=model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0,
        key="chat_llm_direction",
    )
    st.divider()

    current_desc = ""
    try:
        with open(desc_path, "r", encoding="utf-8") as f:
            current_desc = f.read()
    except FileNotFoundError:
        pass

    def _get_desc():
        msgs = st.session_state.get("chat_direction", [])
        for m in reversed(msgs):
            if m["role"] == "assistant" and m.get("is_revision"):
                return m["content"]
        return current_desc or "（暂无研究方向，请通过对话生成）"

    def _save_desc(content):
        yz.set_data_description(content)
        st.session_state["chat_direction"] = []
        st.success("✅ 研究方向已保存！")

    if not current_desc:
        st.info("💡 可以先输入一个关键词或领域，让 AI 帮你扩展成完整的研究方向描述。例如：「我想研究图神经网络在推荐系统中的应用」")

    chat_section(
        yz=yz,
        page_key="direction",
        content_type="研究方向",
        get_current_content=_get_desc,
        on_save=_save_desc,
        llm_model=chat_llm,
    )
