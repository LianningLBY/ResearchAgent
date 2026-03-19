"""论文审阅页。"""
import os
import streamlit as st
from yanzhi import models
from utils import show_markdown_file, stream_to_streamlit

yz = st.session_state.yz

st.title("✅ 论文审阅")
st.markdown("对已生成的论文进行 AI 同行评审，输出详细审稿意见和改进建议。")
st.divider()

# 检查论文是否存在
pdffile = os.path.join(yz.project_dir, "paper", "paper_v4_final.pdf")
if not os.path.exists(pdffile):
    st.warning("⚠️ 尚未生成论文，请先前往「论文生成」完成论文写作。")
else:
    st.success("✅ 检测到已生成的论文，可以开始审阅。")

model_keys = list(models.keys())
default_idx = (model_keys.index("gemini-2.5-flash")
               if "gemini-2.5-flash" in model_keys else 0)

with st.expander("⚙️ 审阅选项"):
    llm_model = st.selectbox("LLM 模型", model_keys, index=default_idx, key="llm_referee")
    verbose   = st.toggle("显示运行日志", value=True, key="verbose_referee")

if "referee_running" not in st.session_state:
    st.session_state.referee_running = False

col1, col2 = st.columns([1, 1])
with col1:
    press = st.button("🔍 开始审阅", type="primary", key="btn_referee",
                      disabled=st.session_state.referee_running)
with col2:
    stop = st.button("⏹ 停止", type="secondary", key="stop_referee",
                     disabled=not st.session_state.referee_running)

if press and not st.session_state.referee_running:
    st.session_state.referee_running = True
    st.rerun()
if stop and st.session_state.referee_running:
    st.session_state.referee_running = False
    st.warning("已停止操作。")
    st.rerun()

if st.session_state.referee_running:
    with st.spinner("正在审阅论文…", show_time=True):
        log_box = st.empty()
        with stream_to_streamlit(log_box):
            try:
                yz.referee(llm=llm_model, verbose=verbose)
                if st.session_state.get("referee_running"):
                    st.success("✅ 审阅完成！")
            except FileNotFoundError:
                st.error("未找到论文文件，请先完成「论文生成」。")
            except Exception as e:
                st.error(f"错误：{e}")
            finally:
                st.session_state.referee_running = False

st.divider()
st.subheader("审稿报告")
try:
    show_markdown_file(os.path.join(yz.project_dir, "input_files", "referee.md"), label="审稿报告")
except FileNotFoundError:
    st.caption("尚未生成审稿报告。")
