"""论文生成页。"""
import os
import streamlit as st
from yanzhi import Journal, models
from utils import create_zip_in_memory

yz = st.session_state.yz

st.title("📝 论文生成")
st.markdown("根据研究方法和实验结果，自动生成中文学术论文（LaTeX 格式，支持 ctex 中文排版）。")
st.divider()

model_keys = list(models.keys())
default_idx = model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0

with st.expander("⚙️ 论文生成选项"):
    llm_model = st.selectbox("LLM 模型", model_keys, index=default_idx, key="llm_paper")

    journal_options = {
        "标准中文格式（ctexart，单栏）": Journal.NONE,
        "中国知网双栏格式（CNKI）":     Journal.CNKI,
        "IEEE 双栏（含中文 ctex 支持）": Journal.IEEE,
    }
    selected_label = st.selectbox("排版格式", list(journal_options.keys()), key="journal_select")
    selected_journal = journal_options[selected_label]

    add_citations = st.toggle("自动添加参考文献（需要 Perplexity API Key）",
                              value=False, key="citations_toggle")
    writer = st.text_input(
        "作者身份（影响写作风格）",
        value="资深科研人员",
        placeholder="例如：计算机视觉研究员、生物信息学专家",
        key="writer_type",
    )

if "paper_running" not in st.session_state:
    st.session_state.paper_running = False

col1, col2 = st.columns([1, 1])
with col1:
    press = st.button("🚀 生成论文", type="primary", key="btn_paper",
                      disabled=st.session_state.paper_running)
with col2:
    stop = st.button("⏹ 停止", type="secondary", key="stop_paper",
                     disabled=not st.session_state.paper_running)

if press and not st.session_state.paper_running:
    st.session_state.paper_running = True
    st.rerun()
if stop and st.session_state.paper_running:
    st.session_state.paper_running = False
    st.warning("已停止操作。")
    st.rerun()

if st.session_state.paper_running:
    with st.spinner("正在生成论文，这可能需要几分钟…", show_time=True):
        try:
            yz.get_paper(journal=selected_journal, llm=llm_model,
                         writer=writer, add_citations=add_citations)
            if st.session_state.get("paper_running"):
                st.success("✅ 论文生成完成！")
                st.balloons()
                st.info("👉 下一步：前往「论文审阅」获取 AI 审稿意见")
        except Exception as e:
            st.error(f"错误：{e}")
        finally:
            st.session_state.paper_running = False

st.divider()

# 下载 LaTeX 和 PDF
col_a, col_b = st.columns(2)
with col_a:
    texfile = os.path.join(yz.project_dir, "paper", "paper_v4_final.tex")
    if os.path.exists(texfile):
        paper_zip = create_zip_in_memory(os.path.join(yz.project_dir, "paper"))
        st.download_button("⬇️ 下载 LaTeX 源文件", data=paper_zip,
                           file_name="paper.zip", mime="application/zip",
                           use_container_width=True)
    else:
        st.caption("LaTeX 文件尚未生成。")

with col_b:
    pdffile = os.path.join(yz.project_dir, "paper", "paper_v4_final.pdf")
    if os.path.exists(pdffile):
        with open(pdffile, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("⬇️ 下载 PDF", data=pdf_bytes,
                           file_name="paper.pdf", mime="application/octet-stream",
                           use_container_width=True)
    else:
        st.caption("PDF 尚未生成。")

# PDF 预览
pdffile = os.path.join(yz.project_dir, "paper", "paper_v4_final.pdf")
if os.path.exists(pdffile):
    st.divider()
    st.subheader("论文预览")
    try:
        from streamlit_pdf_viewer import pdf_viewer
        pdf_viewer(pdffile)
    except ImportError:
        st.info("安装 `streamlit-pdf-viewer` 可在页面内预览 PDF：`pip install streamlit-pdf-viewer`")
