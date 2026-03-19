"""研究方法生成页。支持版本历史、改进反馈和对话修改。"""
import os
import streamlit as st
from yanzhi import models
from utils import show_markdown_file, stream_to_streamlit, show_history, chat_section

yz = st.session_state.yz
method_path = os.path.join(yz.project_dir, "input_files", "methods.md")
has_method  = os.path.exists(method_path)

st.title("🔬 研究方法")

tab_gen, tab_chat = st.tabs(["🚀 生成", "💬 对话修改"])

with tab_gen:
    st.markdown("根据研究想法生成详细的实验方案和技术路线。每次生成前自动备份。")

    model_keys = list(models.keys())
    default_idx = model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0

    with st.expander("⚙️ 生成选项"):
        llm_model = st.selectbox("LLM 模型", model_keys, index=default_idx, key="llm_method")
        verbose   = st.toggle("显示运行日志", value=True, key="verbose_method")

    feedback_text = ""
    if has_method:
        with st.expander("💬 本次生成的改进方向（可选）"):
            st.caption("填写对上一版方法的不满意之处。也可直接在「对话修改」标签页逐步调整。")
            feedback_text = st.text_area(
                "改进建议",
                placeholder="例如：\n- 需要更详细的数据预处理步骤\n- 增加消融实验设计\n- 补充基线模型的选择依据",
                key="method_feedback",
                height=100,
            )

    if "method_running" not in st.session_state:
        st.session_state.method_running = False

    col1, col2 = st.columns([1, 1])
    with col1:
        label = "🔄 基于反馈重新生成" if (has_method and feedback_text.strip()) else "🚀 生成研究方法"
        press = st.button(label, type="primary", key="btn_method",
                          disabled=st.session_state.method_running)
    with col2:
        stop = st.button("⏹ 停止", type="secondary", key="stop_method",
                         disabled=not st.session_state.method_running)

    if press and not st.session_state.method_running:
        if feedback_text.strip():
            desc_path = os.path.join(yz.project_dir, "input_files", "data_description.md")
            try:
                with open(desc_path, "r", encoding="utf-8") as f:
                    orig = f.read()
                with open(desc_path, "w", encoding="utf-8") as f:
                    f.write(orig + f"\n\n---\n### 对上一版研究方法的改进反馈\n{feedback_text.strip()}")
                st.session_state["_method_orig_desc"] = orig
            except FileNotFoundError:
                pass
        st.session_state.method_running = True
        st.session_state["chat_methods"] = []
        st.rerun()

    if stop and st.session_state.method_running:
        st.session_state.method_running = False
        st.warning("已停止。")
        st.rerun()

    if st.session_state.method_running:
        with st.spinner("正在生成研究方法…", show_time=True):
            log_box = st.empty()
            with stream_to_streamlit(log_box):
                try:
                    yz.get_method(llm=llm_model, verbose=verbose)
                    if st.session_state.get("method_running"):
                        st.success("✅ 生成完成！可在「对话修改」标签页继续调整。")
                except Exception as e:
                    st.error(f"错误：{e}")
                finally:
                    st.session_state.method_running = False
                    if "_method_orig_desc" in st.session_state:
                        desc_path = os.path.join(yz.project_dir, "input_files", "data_description.md")
                        with open(desc_path, "w", encoding="utf-8") as f:
                            f.write(st.session_state.pop("_method_orig_desc"))

    st.divider()
    show_history(yz, "methods", on_load=lambda c: yz.set_method(c))

    with st.expander("✏️ 手动上传"):
        up = st.file_uploader("上传 Markdown", key="upload_method", accept_multiple_files=False)
        if up:
            yz.set_method(up.read().decode("utf-8"))
            st.success("✅ 已上传！")
            st.rerun()

    st.divider()
    st.subheader("当前研究方法")
    try:
        show_markdown_file(method_path, label="研究方法")
    except FileNotFoundError:
        st.caption("尚未生成研究方法。")

with tab_chat:
    if not has_method:
        st.info("请先在「生成」标签页生成研究方法，再进行对话修改。")
    else:
        model_keys = list(models.keys())
        chat_llm = st.selectbox(
            "对话模型",
            model_keys,
            index=model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0,
            key="chat_llm_methods",
        )
        st.divider()

        def _get_current_method():
            msgs = st.session_state.get("chat_methods", [])
            for m in reversed(msgs):
                if m["role"] == "assistant" and m.get("is_revision"):
                    return m["content"]
            try:
                with open(method_path, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return ""

        def _save_method(content):
            yz.set_method(content)
            st.session_state["chat_methods"] = []

        chat_section(
            yz=yz,
            page_key="methods",
            content_type="研究方法",
            get_current_content=_get_current_method,
            on_save=_save_method,
            llm_model=chat_llm,
        )
