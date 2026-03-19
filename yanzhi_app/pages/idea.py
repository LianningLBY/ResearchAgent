"""研究想法生成页。支持版本历史、改进反馈和对话修改。"""
import os
import streamlit as st
from yanzhi import models
from utils import show_markdown_file, stream_to_streamlit, show_history, chat_section

yz = st.session_state.yz
idea_path = os.path.join(yz.project_dir, "input_files", "idea.md")
has_idea  = os.path.exists(idea_path)

st.title("💡 研究想法")

# ---------------------------------------------------------------------------
# 标签页：生成 / 对话修改
# ---------------------------------------------------------------------------
tab_gen, tab_chat = st.tabs(["🚀 生成", "💬 对话修改"])

with tab_gen:
    st.markdown("通过 IdeaMaker/IdeaHater 对抗迭代生成创新研究想法。每次生成前自动备份上一版本。")

    # 文献背景提示
    lit_path = os.path.join(yz.project_dir, "input_files", "literature.md")
    if os.path.exists(lit_path):
        with st.expander("📚 已有文献背景（将辅助想法生成）"):
            with open(lit_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content[:500] + ("…" if len(content) > 500 else ""))

    model_keys = list(models.keys())
    default_idx = model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0

    with st.expander("⚙️ 生成选项"):
        llm_model  = st.selectbox("LLM 模型", model_keys, index=default_idx, key="llm_idea")
        iterations = st.slider("对抗迭代轮数", min_value=2, max_value=10, value=4, key="iter_idea")
        verbose    = st.toggle("显示运行日志", value=True, key="verbose_idea")

    # 改进反馈（有已有想法时显示）
    feedback_text = ""
    if has_idea:
        with st.expander("💬 本次生成的改进方向（可选）"):
            st.caption("填写对上一版的不满意之处，AI 将据此生成新版本。也可直接在「对话修改」标签页逐步调整。")
            feedback_text = st.text_area(
                "改进建议",
                placeholder="例如：\n- 想法太宽泛，请聚焦在小样本场景\n- 缺少可量化的评估指标\n- 需要更明确的创新点",
                key="idea_feedback",
                height=100,
            )

    if "idea_running" not in st.session_state:
        st.session_state.idea_running = False

    col1, col2 = st.columns([1, 1])
    with col1:
        label = "🔄 基于反馈重新生成" if (has_idea and feedback_text.strip()) else "🚀 生成研究想法"
        press = st.button(label, type="primary", key="btn_idea",
                          disabled=st.session_state.idea_running)
    with col2:
        stop = st.button("⏹ 停止", type="secondary", key="stop_idea",
                         disabled=not st.session_state.idea_running)

    if press and not st.session_state.idea_running:
        if feedback_text.strip():
            desc_path = os.path.join(yz.project_dir, "input_files", "data_description.md")
            try:
                with open(desc_path, "r", encoding="utf-8") as f:
                    orig = f.read()
                with open(desc_path, "w", encoding="utf-8") as f:
                    f.write(orig + f"\n\n---\n### 对上一版研究想法的改进反馈\n{feedback_text.strip()}")
                st.session_state["_idea_orig_desc"] = orig
            except FileNotFoundError:
                pass
        st.session_state.idea_running = True
        # 新一轮生成时清空对话记录
        st.session_state["chat_idea"] = []
        st.rerun()

    if stop and st.session_state.idea_running:
        st.session_state.idea_running = False
        st.warning("已停止。")
        st.rerun()

    if st.session_state.idea_running:
        with st.spinner("正在生成研究想法…", show_time=True):
            log_box = st.empty()
            with stream_to_streamlit(log_box):
                try:
                    yz.get_idea(llm=llm_model, iterations=iterations, verbose=verbose)
                    if st.session_state.get("idea_running"):
                        st.success("✅ 生成完成！可在「对话修改」标签页继续调整。")
                except Exception as e:
                    st.error(f"错误：{e}")
                finally:
                    st.session_state.idea_running = False
                    if "_idea_orig_desc" in st.session_state:
                        desc_path = os.path.join(yz.project_dir, "input_files", "data_description.md")
                        with open(desc_path, "w", encoding="utf-8") as f:
                            f.write(st.session_state.pop("_idea_orig_desc"))

    st.divider()
    show_history(yz, "idea", on_load=lambda c: yz.set_idea(c))

    with st.expander("✏️ 手动输入或上传"):
        manual = st.text_area("直接输入研究想法", key="manual_idea_text", height=120)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("保存", key="save_manual_idea") and manual.strip():
                yz.set_idea(manual)
                st.success("✅ 已保存！")
                st.rerun()
        with col_b:
            up = st.file_uploader("上传 Markdown", key="upload_idea", accept_multiple_files=False)
            if up:
                yz.set_idea(up.read().decode("utf-8"))
                st.success("✅ 已上传！")
                st.rerun()

    st.divider()
    st.subheader("当前研究想法")
    try:
        show_markdown_file(idea_path, label="研究想法")
    except FileNotFoundError:
        st.caption("尚未生成研究想法。")

# ---------------------------------------------------------------------------
# 对话修改标签页
# ---------------------------------------------------------------------------
with tab_chat:
    if not has_idea:
        st.info("请先在「生成」标签页生成研究想法，再进行对话修改。")
    else:
        # 模型选择放在对话界面顶部
        model_keys = list(models.keys())
        chat_llm = st.selectbox(
            "对话模型",
            model_keys,
            index=model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0,
            key="chat_llm_idea",
        )
        st.divider()

        def _get_current_idea():
            # 优先用最后一条 AI 回复（最新草稿），否则用文件
            msgs = st.session_state.get("chat_idea", [])
            for m in reversed(msgs):
                if m["role"] == "assistant" and m.get("is_revision"):
                    return m["content"]
            try:
                with open(idea_path, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return ""

        def _save_idea(content):
            yz.set_idea(content)
            # 清空对话，因为已应用到文件
            st.session_state["chat_idea"] = []

        chat_section(
            yz=yz,
            page_key="idea",
            content_type="研究想法",
            get_current_content=_get_current_idea,
            on_save=_save_idea,
            llm_model=chat_llm,
        )
