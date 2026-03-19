from pathlib import Path
import streamlit as st
from yanzhi import YanZhi, Journal, models

from utils import show_markdown_file, create_zip_in_memory, stream_to_streamlit

# 停止按钮样式
_STOP_BTN_CSS = """
<style>
div[data-testid="column"]:nth-of-type(2) button {
    border: 2px solid #ff4444 !important;
    color: #ff4444 !important;
}
div[data-testid="column"]:nth-of-type(2) button:hover {
    background-color: #ff4444 !important;
    color: white !important;
}
</style>
"""


def _run_button_row(key: str):
    """返回 (press, stop) 两个按钮状态，并管理 session_state。"""
    running_key = f"{key}_running"
    if running_key not in st.session_state:
        st.session_state[running_key] = False

    st.markdown(_STOP_BTN_CSS, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        press = st.button("开始生成", type="primary", key=f"btn_{key}",
                          disabled=st.session_state[running_key])
    with col2:
        stop = st.button("停止", type="secondary", key=f"stop_{key}",
                         disabled=not st.session_state[running_key])

    if press and not st.session_state[running_key]:
        st.session_state[running_key] = True
        st.rerun()
    if stop and st.session_state[running_key]:
        st.session_state[running_key] = False
        st.warning("已停止操作。")
        st.rerun()

    return st.session_state[running_key]


# ---------------------------------------------------------------------------
# 各模块组件
# ---------------------------------------------------------------------------

def description_comp(yz: YanZhi) -> None:
    st.header("输入描述")

    current_description = ""
    try:
        with open(yz.project_dir + "/input_files/data_description.md", "r", encoding="utf-8") as f:
            current_description = f.read()
    except FileNotFoundError:
        pass

    data_descr = st.text_area(
        "请描述本项目使用的数据和工具，也可以包含计算资源的说明。",
        placeholder="例如：使用 sklearn 和 pandas 分析存储在 /path/to/data.csv 中的实验数据。"
                    "该数据包含来自粒子探测器的时间序列测量值。",
        value=current_description,
        key="data_descr",
        height=120,
    )

    uploaded_file = st.file_uploader(
        "或者上传一个包含数据描述的 Markdown 文件。",
        accept_multiple_files=False,
    )

    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        yz.set_data_description(content)

    if data_descr:
        yz.set_data_description(data_descr)

    st.markdown("### 当前数据描述")
    try:
        show_markdown_file(yz.project_dir + "/input_files/data_description.md",
                           label="数据描述")
    except FileNotFoundError:
        st.write("尚未设置数据描述。")


def idea_comp(yz: YanZhi) -> None:
    st.header("研究想法")
    st.write("根据数据描述，生成一个具有创新性的研究想法（IdeaMaker/IdeaHater 对抗迭代）。")

    model_keys = list(models.keys())
    default_idx = model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0

    with st.expander("生成选项"):
        llm_model = st.selectbox("选择 LLM 模型", model_keys, index=default_idx, key="llm_idea")
        iterations = st.slider("迭代轮数", min_value=2, max_value=10, value=4, key="iter_idea")
        verbose = st.toggle("流式输出日志", value=True, key="verbose_idea")

    running = _run_button_row("idea")

    if running:
        with st.spinner("正在生成研究想法…", show_time=True):
            log_box = st.empty()
            with stream_to_streamlit(log_box):
                try:
                    yz.get_idea(llm=llm_model, iterations=iterations, verbose=verbose)
                    if st.session_state.get("idea_running"):
                        st.success("研究想法生成完成！")
                except Exception as e:
                    st.error(f"错误：{e}")
                finally:
                    st.session_state["idea_running"] = False

    uploaded_file = st.file_uploader("或上传已有研究想法文件（Markdown）", accept_multiple_files=False,
                                     key="upload_idea")
    if uploaded_file:
        yz.set_idea(uploaded_file.read().decode("utf-8"))

    try:
        show_markdown_file(yz.project_dir + "/input_files/idea.md",
                           extra_format=True, label="研究想法")
    except FileNotFoundError:
        st.write("尚未生成或上传研究想法。")


def method_comp(yz: YanZhi) -> None:
    st.header("研究方法")
    st.write("根据数据描述和研究想法，生成研究方法论。")

    model_keys = list(models.keys())
    default_idx = model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0

    with st.expander("生成选项"):
        llm_model = st.selectbox("选择 LLM 模型", model_keys, index=default_idx, key="llm_method")
        verbose = st.toggle("流式输出日志", value=True, key="verbose_method")

    running = _run_button_row("method")

    if running:
        with st.spinner("正在生成研究方法…", show_time=True):
            log_box = st.empty()
            with stream_to_streamlit(log_box):
                try:
                    yz.get_method(llm=llm_model, verbose=verbose)
                    if st.session_state.get("method_running"):
                        st.success("研究方法生成完成！")
                except Exception as e:
                    st.error(f"错误：{e}")
                finally:
                    st.session_state["method_running"] = False

    uploaded_file = st.file_uploader("或上传已有方法文件（Markdown）",
                                     accept_multiple_files=False, key="upload_method")
    if uploaded_file:
        yz.set_method(uploaded_file.read().decode("utf-8"))

    try:
        show_markdown_file(yz.project_dir + "/input_files/methods.md", label="研究方法")
    except FileNotFoundError:
        st.write("尚未生成或上传研究方法。")


def results_comp(yz: YanZhi) -> None:
    st.header("实验结果")
    st.write("上传实验数据分析结果和图表。（本系统不自动执行代码，请在本地运行后上传结果。）")

    uploaded_files = st.file_uploader(
        "上传结果 Markdown 文件和/或图表（PNG/JPG）",
        accept_multiple_files=True,
        key="upload_results",
    )
    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith(".md"):
                yz.set_results(file.read().decode("utf-8"))
            else:
                # 保存图片到 plots 文件夹
                import os
                plots_dir = yz.project_dir + "/input_files/plots"
                os.makedirs(plots_dir, exist_ok=True)
                with open(os.path.join(plots_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())
        st.success("上传成功！")

    # 显示图表
    plots = list(Path(yz.project_dir + "/input_files/plots").glob("*"))
    if plots:
        cols = st.columns(min(len(plots), 4))
        for i, plot in enumerate(plots):
            with cols[i % 4]:
                st.image(str(plot), caption=plot.name)

        plots_zip = create_zip_in_memory(yz.project_dir + "/input_files/plots")
        st.download_button(
            label="下载所有图表",
            data=plots_zip,
            file_name="plots.zip",
            mime="application/zip",
            icon=":material/download:",
        )
    else:
        st.write("尚未上传图表。")

    try:
        show_markdown_file(yz.project_dir + "/input_files/results.md", label="实验结果摘要")
    except FileNotFoundError:
        st.write("尚未上传实验结果。")


def paper_comp(yz: YanZhi) -> None:
    st.header("论文生成")
    st.write("根据实验结果，生成中文学术论文（LaTeX 格式，支持 ctex 中文排版）。")

    model_keys = list(models.keys())
    default_idx = model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0

    with st.expander("论文生成选项"):
        llm_model = st.selectbox("选择 LLM 模型", model_keys, index=default_idx, key="llm_paper")

        journal_options = {
            "标准中文格式（ctexart）": Journal.NONE,
            "中国知网双栏格式（CNKI）": Journal.CNKI,
            "IEEE 双栏（含中文支持）": Journal.IEEE,
        }
        selected_journal_label = st.selectbox(
            "选择期刊/排版格式", list(journal_options.keys()), index=0, key="journal_select"
        )
        selected_journal = journal_options[selected_journal_label]

        add_citations = st.toggle("自动添加参考文献（需要 Perplexity API Key）",
                                  value=False, key="citations_toggle")

        writer = st.text_input(
            "作者身份描述（影响写作风格）",
            placeholder="例如：资深物理学家、生物信息学研究员",
            value="资深科研人员",
            key="writer_type",
        )

    running = _run_button_row("paper")

    if running:
        with st.spinner("正在生成论文，请稍候…", show_time=True):
            try:
                yz.get_paper(
                    journal=selected_journal,
                    llm=llm_model,
                    writer=writer,
                    add_citations=add_citations,
                )
                if st.session_state.get("paper_running"):
                    st.success("论文生成完成！")
                    st.balloons()
            except Exception as e:
                st.error(f"错误：{e}")
            finally:
                st.session_state["paper_running"] = False

    # 下载 LaTeX
    try:
        texfile = yz.project_dir + "/paper/paper_v4_final.tex"
        with open(texfile, "r", encoding="utf-8"):
            pass
        paper_zip = create_zip_in_memory(yz.project_dir + "/paper")
        st.download_button(
            label="下载 LaTeX 源文件",
            data=paper_zip,
            file_name="paper.zip",
            mime="application/zip",
            icon=":material/download:",
        )
    except FileNotFoundError:
        st.write("LaTeX 文件尚未生成。")

    # 下载 PDF 并预览
    try:
        pdffile = yz.project_dir + "/paper/paper_v4_final.pdf"
        with open(pdffile, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="下载 PDF",
            data=pdf_bytes,
            file_name="paper.pdf",
            mime="application/octet-stream",
            icon=":material/download:",
        )
        try:
            from streamlit_pdf_viewer import pdf_viewer
            pdf_viewer(pdffile)
        except ImportError:
            st.info("安装 streamlit-pdf-viewer 可在页面内预览 PDF。")
    except FileNotFoundError:
        st.write("PDF 文件尚未生成。")


def check_idea_comp(yz: YanZhi) -> None:
    st.header("文献查新")
    st.write("通过 Semantic Scholar 检索相关文献，验证研究想法的新颖性。")

    model_keys = list(models.keys())
    default_idx = model_keys.index("gemini-2.0-flash") if "gemini-2.0-flash" in model_keys else 0

    with st.expander("查新选项"):
        llm_model = st.selectbox("选择 LLM 模型", model_keys, index=default_idx, key="llm_literature")
        max_iter = st.slider("最大检索轮数", min_value=3, max_value=15, value=7, key="lit_max_iter")
        verbose = st.toggle("流式输出日志", value=True, key="verbose_literature")

    try:
        yz.set_idea()
        st.markdown("### 当前研究想法")
        st.write(yz.research.idea)

        running = _run_button_row("literature")

        if running:
            with st.spinner("正在检索相关文献…", show_time=True):
                log_box = st.empty()
                with stream_to_streamlit(log_box):
                    try:
                        result = yz.check_idea(llm=llm_model, max_iterations=max_iter, verbose=verbose)
                        st.write(result)
                        if st.session_state.get("literature_running"):
                            st.success("文献查新完成！")
                    except Exception as e:
                        st.error(f"错误：{e}")
                    finally:
                        st.session_state["literature_running"] = False

    except FileNotFoundError:
        st.write("请先生成研究想法。")

    try:
        show_markdown_file(yz.project_dir + "/input_files/literature.md", label="文献查新报告")
    except FileNotFoundError:
        pass


def referee_comp(yz: YanZhi) -> None:
    st.header("论文审阅")
    st.write("对已生成的论文进行同行评审，输出详细的审稿意见。")

    model_keys = list(models.keys())
    default_idx = (model_keys.index("gemini-2.5-flash")
                   if "gemini-2.5-flash" in model_keys else 0)

    with st.expander("审阅选项"):
        llm_model = st.selectbox("选择 LLM 模型", model_keys, index=default_idx, key="llm_referee")
        verbose = st.toggle("流式输出日志", value=True, key="verbose_referee")

    running = _run_button_row("referee")

    if running:
        with st.spinner("正在审阅论文…", show_time=True):
            log_box = st.empty()
            with stream_to_streamlit(log_box):
                try:
                    yz.referee(llm=llm_model, verbose=verbose)
                    if st.session_state.get("referee_running"):
                        st.success("审阅完成！")
                except FileNotFoundError:
                    st.error("未找到论文文件，请先运行「论文生成」。")
                except Exception as e:
                    st.error(f"错误：{e}")
                finally:
                    st.session_state["referee_running"] = False

    try:
        show_markdown_file(yz.project_dir + "/input_files/referee.md", label="审稿报告")
    except FileNotFoundError:
        st.write("尚未生成审稿报告。")
