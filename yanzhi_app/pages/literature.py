"""文献调研页：支持在生成想法前后搜索文献，包含历史版本。"""
import os
import streamlit as st
from yanzhi import models
from utils import show_markdown_file, stream_to_streamlit, show_history

yz = st.session_state.yz

st.title("📚 文献调研")
st.markdown("搜索相关文献，了解领域现状。可在生成研究想法**之前**使用（探索方向），也可在**之后**使用（验证新颖性）。")
st.divider()

model_keys = list(models.keys())
default_idx = model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0

with st.expander("⚙️ 搜索选项"):
    llm_model = st.selectbox("LLM 模型", model_keys, index=default_idx, key="llm_literature")
    max_iter  = st.slider("最大检索轮数", min_value=3, max_value=15, value=7, key="lit_max_iter")
    verbose   = st.toggle("显示运行日志", value=True, key="verbose_literature")

st.divider()

idea_path = os.path.join(yz.project_dir, "input_files", "idea.md")
desc_path  = os.path.join(yz.project_dir, "input_files", "data_description.md")
has_desc = os.path.exists(desc_path)

idea_content = ""
if os.path.exists(idea_path):
    try:
        with open(idea_path, "r", encoding="utf-8") as f:
            idea_content = f.read()
    except FileNotFoundError:
        idea_content = ""
has_idea = bool(idea_content.strip())

desc_newer_than_idea = False
if has_desc and has_idea:
    try:
        desc_mtime = os.path.getmtime(desc_path)
        idea_mtime = os.path.getmtime(idea_path)
        desc_newer_than_idea = desc_mtime > idea_mtime
    except OSError:
        pass

use_stale_idea = st.session_state.get("literature_use_stale_idea", False)
if desc_newer_than_idea and not use_stale_idea:
    st.warning("⚠️ 研究方向已更新，但研究想法仍是旧版本。建议先清空想法并基于新方向重新检索。")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🧹 清空旧想法并切换为基于研究方向检索", key="clear_stale_idea"):
            try:
                yz._backup_file("idea.md")
            except Exception:
                pass
            try:
                with open(idea_path, "w", encoding="utf-8") as f:
                    f.write("")
                st.success("✅ 已清空旧想法，现在可基于研究方向生成检索关键词。")
            except Exception as e:
                st.error(f"清空失败：{e}")
            st.rerun()
    with col_b:
        if st.button("继续使用旧想法", key="use_stale_idea"):
            st.session_state["literature_use_stale_idea"] = True
            st.rerun()
    has_idea = False

if has_idea:
    st.success("✅ 已有研究想法，将基于想法进行文献查新。")
    if st.button("🧹 清空研究想法并切换为基于研究方向检索", key="clear_idea_for_lit"):
        try:
            yz._backup_file("idea.md")
        except Exception:
            pass
        try:
            with open(idea_path, "w", encoding="utf-8") as f:
                f.write("")
            st.success("✅ 已清空研究想法，现在可基于研究方向生成检索关键词。")
        except Exception as e:
            st.error(f"清空失败：{e}")
        st.rerun()
    with st.expander("当前研究想法"):
        st.markdown(idea_content)
elif has_desc:
    st.info("📖 尚未生成研究想法。请输入初步搜索关键词用于文献检索。")
    try:
        with open(desc_path, "r", encoding="utf-8") as f:
            desc_content = f.read()
        st.caption("当前研究方向：")
        st.markdown(f"> {desc_content[:200]}{'…' if len(desc_content) > 200 else ''}")
    except FileNotFoundError:
        pass

    if st.button("🧠 基于研究方向自动生成检索关键词", key="auto_prelim"):
        with st.spinner("正在生成检索关键词…", show_time=True):
            try:
                generated = yz.suggest_literature_query(llm=llm_model)
                st.session_state["prelim_idea_input"] = generated
                st.success("✅ 已生成检索关键词，请确认后保存为临时想法。")
            except Exception as e:
                st.error(f"生成失败：{e}")

    prelim_idea = st.text_area(
        "请输入初步搜索方向（将作为临时想法）",
        placeholder="例如：小样本医学图像分类、域适应、Transformer",
        key="prelim_idea_input",
        height=160,
    )
    if prelim_idea and st.button("保存为临时想法", key="save_prelim"):
        yz.set_idea(prelim_idea)
        st.success("✅ 临时想法已保存，可以开始文献搜索。")
        st.rerun()
else:
    st.warning("⚠️ 请先前往「研究方向」填写研究方向。")

st.divider()

if "literature_running" not in st.session_state:
    st.session_state.literature_running = False

can_search = has_idea
col1, col2 = st.columns([1, 1])
with col1:
    press = st.button("🔍 开始文献调研", type="primary", key="btn_literature",
                      disabled=st.session_state.literature_running or not can_search)
with col2:
    stop = st.button("⏹ 停止", type="secondary", key="stop_literature",
                     disabled=not st.session_state.literature_running)

if press and not st.session_state.literature_running:
    st.session_state.literature_running = True
    st.rerun()
if stop and st.session_state.literature_running:
    st.session_state.literature_running = False
    st.warning("已停止操作。")
    st.rerun()

if st.session_state.literature_running:
    with st.spinner("正在检索相关文献，请稍候…", show_time=True):
        log_box = st.empty()
        with stream_to_streamlit(log_box):
            try:
                result = yz.check_idea(llm=llm_model, max_iterations=max_iter, verbose=verbose)
                if st.session_state.get("literature_running"):
                    failed = (not result
                              or "错误" in result
                              or "失败" in result
                              or result == "查新过程中发生错误"
                              or result == "未找到文献查新结果文件")
                    if failed:
                        st.error(f"❌ 文献调研失败：{result}")
                        st.info("💡 常见原因：LLM 返回格式异常、网络超时。请重试，或尝试换一个模型。")
                    else:
                        st.success("✅ 文献调研完成！")
                        if not has_idea:
                            st.info("👉 现在可以前往「研究想法」生成基于文献的研究想法。")
            except Exception as e:
                st.error(f"错误：{e}")
            finally:
                st.session_state.literature_running = False

st.divider()
show_history(yz, "literature", on_load=lambda content: yz.set_method(content))

st.divider()
st.subheader("文献调研报告")
lit_path = os.path.join(yz.project_dir, "input_files", "literature.md")
try:
    show_markdown_file(lit_path, label="文献报告")
except FileNotFoundError:
    st.caption("尚未进行文献调研。")
