"""项目概览页：显示当前项目进度和历史版本统计。"""
import os
import streamlit as st
from constants import PROJECTS_ROOT

yz = st.session_state.yz
current = getattr(st.session_state, "current_project", os.path.basename(yz.project_dir))

st.title(f"📊 {current}")
st.divider()

# ---------------------------------------------------------------------------
# 研究进度
# ---------------------------------------------------------------------------
def _file(name):
    return os.path.exists(os.path.join(yz.project_dir, "input_files", name))

def _paper():
    return os.path.exists(os.path.join(yz.project_dir, "paper", "paper_v4_final.pdf"))

steps = [
    ("🎯 研究方向", _file("data_description.md")),
    ("📚 文献调研", _file("literature.md")),
    ("💡 研究想法", _file("idea.md")),
    ("🔬 研究方法", _file("methods.md")),
    ("📈 实验结果", _file("results.md")),
    ("📝 论文生成", _paper()),
    ("✅ 论文审阅", _file("referee.md")),
]

st.subheader("研究进度")
cols = st.columns(len(steps))
done_count = sum(1 for _, d in steps if d)
for col, (label, done) in zip(cols, steps):
    with col:
        if done:
            st.success(label)
        else:
            st.info(label)

progress = done_count / len(steps)
st.progress(progress, text=f"完成度 {done_count}/{len(steps)}")

# ---------------------------------------------------------------------------
# 历史版本统计
# ---------------------------------------------------------------------------
st.divider()
st.subheader("迭代记录")
history_types = [
    ("💡 想法",   "idea"),
    ("📚 文献",   "literature"),
    ("🔬 方法",   "methods"),
]
h_cols = st.columns(len(history_types))
for col, (label, ftype) in zip(h_cols, history_types):
    history = yz.list_history(ftype)
    with col:
        if history:
            st.metric(label, f"{len(history)} 个版本")
            try:
                from datetime import datetime
                ts = history[0]["timestamp"]
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                st.caption(f"最新：{dt.strftime('%m-%d %H:%M')}")
            except Exception:
                pass
        else:
            st.metric(label, "尚未生成")

# ---------------------------------------------------------------------------
# 所有项目列表
# ---------------------------------------------------------------------------
st.divider()
st.subheader("所有项目")

if not os.path.exists(PROJECTS_ROOT):
    st.caption("暂无其他项目。")
else:
    projects = sorted([d for d in os.listdir(PROJECTS_ROOT)
                       if os.path.isdir(os.path.join(PROJECTS_ROOT, d))])
    if not projects:
        st.caption("暂无项目。")
    else:
        for proj in projects:
            proj_dir = os.path.join(PROJECTS_ROOT, proj)
            # 统计完成情况
            done = sum(1 for fname in [
                "input_files/data_description.md", "input_files/idea.md",
                "input_files/methods.md", "paper/paper_v4_final.pdf"
            ] if os.path.exists(os.path.join(proj_dir, fname)))

            badge = "🟢" if done == 4 else ("🟡" if done >= 2 else "⬜")
            label = f"{badge} **{proj}**" + (" ← 当前" if proj == current else "")
            st.markdown(label)
            st.caption(f"  完成 {done}/4 个关键步骤")

# ---------------------------------------------------------------------------
# 推荐工作流
# ---------------------------------------------------------------------------
st.divider()
with st.expander("📖 推荐工作流"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**从研究方向出发（无数据）**

1. 🎯 研究方向 — 描述科学问题
2. 📚 文献调研 — 了解领域现状
3. 💡 研究想法 — 生成创新想法
4. 🔬 研究方法 — 设计实验方案
5. 📈 实验结果 — 上传本地结果
6. 📝 论文生成 — 自动写作
7. ✅ 论文审阅 — AI 审稿
""")
    with col2:
        st.markdown("""
**迭代改进**

- 每次「研究想法」「研究方法」生成前自动备份
- 可在各页面加载历史版本
- 提供改进反馈后重新生成，AI 将参考反馈优化输出
- 不同课题用不同**项目**隔离，互不干扰
""")
