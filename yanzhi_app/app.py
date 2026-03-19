"""
研智 · 中文多智能体科研助手
主入口文件：初始化 YanZhi 对象，设置多页面导航。
"""
import os
import sys
import argparse
import streamlit as st

# 确保无论从哪个目录启动，都能找到 yanzhi 包和 yanzhi_app 内的工具模块
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))          # yanzhi_app/
_ROOT_DIR = os.path.dirname(_APP_DIR)                            # 项目根目录
for _p in (_ROOT_DIR, _APP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# 页面配置（必须第一个调用）
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="研智 · 中文科研助手",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# ---------------------------------------------------------------------------
# 全局样式
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.log-box {
    background-color: #111827;
    color: #d1d5db;
    font-family: monospace;
    padding: 1em;
    border-radius: 8px;
    overflow-y: auto;
    border: 1px solid #4b5563;
    white-space: pre-wrap;
    resize: vertical;
    min-height: 80px;
    max-height: 500px;
}
.step-done { background:#d1fae5; color:#065f46; padding:0.4em 0.8em; border-radius:5px; margin:0.2em 0; }
.step-todo { background:#f3f4f6; color:#9ca3af; padding:0.4em 0.8em; border-radius:5px; margin:0.2em 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 项目管理（deploy 模式下使用临时目录，否则使用 projects/ 下的命名项目）
# ---------------------------------------------------------------------------
from yanzhi import YanZhi
from utils import delete_old_folders, get_project_dir, create_zip_in_memory
from constants import PROJECTS_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--deploy", action="store_true")
deploy = parser.parse_known_args()[0].deploy

if deploy:
    # 部署模式：每个会话独立临时目录
    if "yz" not in st.session_state:
        delete_old_folders()
        project_dir = get_project_dir()
        st.session_state.yz = YanZhi(project_dir=project_dir, clear_project_dir=False)
else:
    # 本地模式：多项目管理
    os.makedirs(PROJECTS_ROOT, exist_ok=True)

    # 列出现有项目
    existing = sorted([
        d for d in os.listdir(PROJECTS_ROOT)
        if os.path.isdir(os.path.join(PROJECTS_ROOT, d))
    ])
    if not existing:
        os.makedirs(os.path.join(PROJECTS_ROOT, "默认项目"), exist_ok=True)
        existing = ["默认项目"]

    if "current_project" not in st.session_state:
        st.session_state.current_project = existing[0]
    elif st.session_state.current_project not in existing:
        st.session_state.current_project = existing[0]

    # ── 侧边栏：项目选择器 ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 项目切换")
        selected = st.selectbox(
            "当前项目",
            existing,
            index=existing.index(st.session_state.current_project),
            key="project_selector",
            label_visibility="collapsed",
        )
        if selected != st.session_state.current_project:
            st.session_state.current_project = selected
            if "yz" in st.session_state:
                del st.session_state.yz
            st.rerun()

        with st.expander("➕ 新建项目"):
            new_name = st.text_input("项目名称", key="new_project_name",
                                     placeholder="例如：医学图像分类研究")
            if st.button("创建", key="btn_create_project", use_container_width=True):
                if new_name.strip():
                    safe = new_name.strip().replace("/", "_").replace("\\", "_")
                    os.makedirs(os.path.join(PROJECTS_ROOT, safe), exist_ok=True)
                    st.session_state.current_project = safe
                    if "yz" in st.session_state:
                        del st.session_state.yz
                    st.rerun()
                else:
                    st.warning("请输入项目名称")

    # ── 初始化 YanZhi ───────────────────────────────────────────────
    if "yz" not in st.session_state:
        project_dir = os.path.join(PROJECTS_ROOT, st.session_state.current_project)
        st.session_state.yz = YanZhi(project_dir=project_dir, clear_project_dir=False)

yz = st.session_state.yz

# ---------------------------------------------------------------------------
# 侧边栏底部：下载按钮
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.caption(f"📁 `{os.path.basename(yz.project_dir)}`")
    try:
        project_zip = create_zip_in_memory(yz.project_dir)
        st.download_button(
            label="⬇️ 下载项目文件",
            data=project_zip,
            file_name=f"{os.path.basename(yz.project_dir)}.zip",
            mime="application/zip",
            use_container_width=True,
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 多页面导航
# ---------------------------------------------------------------------------
overview_page   = st.Page("pages/overview.py",   title="项目概览",   icon="📊", default=True)
direction_page  = st.Page("pages/direction.py",  title="研究方向",   icon="🎯")
literature_page = st.Page("pages/literature.py", title="文献调研",   icon="📚")
idea_page       = st.Page("pages/idea.py",       title="研究想法",   icon="💡")
methods_page    = st.Page("pages/methods.py",    title="研究方法",   icon="🔬")
results_page    = st.Page("pages/results.py",    title="实验结果",   icon="📈")
paper_page      = st.Page("pages/paper.py",      title="论文生成",   icon="📝")
referee_page    = st.Page("pages/referee.py",    title="论文审阅",   icon="✅")
settings_page   = st.Page("pages/settings.py",  title="系统设置",   icon="⚙️")

pg = st.navigation({
    "研究工作台": [
        overview_page,
        direction_page,
        literature_page,
        idea_page,
        methods_page,
        results_page,
        paper_page,
        referee_page,
    ],
    "系统": [settings_page],
})

pg.run()
