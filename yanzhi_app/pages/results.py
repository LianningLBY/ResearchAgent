"""实验结果页：手动上传 或 自动实验（Human-in-the-Loop + 后台线程 + 进度可视化）。"""
import json
import os
import time
import uuid
from pathlib import Path

import streamlit as st
from utils import show_markdown_file, create_zip_in_memory

yz = st.session_state.yz

st.title("📈 实验结果")
st.divider()

tab_auto, tab_manual = st.tabs(["🤖 自动实验", "📤 手动上传"])


# ── 进度可视化组件 ─────────────────────────────────────────────────────────────

def _render_step_progress(module_folder: str) -> None:
    """读取 progress.jsonl，渲染步骤时间轴。"""
    from yanzhi.experiment_agents.progress import read_progress, summarize_progress, STEP_ORDER

    records = read_progress(module_folder)
    if not records:
        st.caption("等待进度信息...")
        return

    summary = summarize_progress(records)
    completed = set(summary["completed"])
    current   = summary["current"]
    errors    = {r["node"] for r in summary["errors"]}
    outer     = summary["outer_iter"]
    inner     = summary["inner_iter"]

    # 迭代轮次提示
    iter_info = []
    if outer > 0:
        iter_info.append(f"外层第 {outer} 轮")
    if inner > 0:
        iter_info.append(f"内层修复 {inner} 次")
    if iter_info:
        st.caption(" · ".join(iter_info))

    # 步骤时间轴
    cols = st.columns(len(STEP_ORDER))
    for i, (node, label) in enumerate(STEP_ORDER):
        with cols[i]:
            if node in errors:
                icon = "❌"
                color = "red"
            elif node in completed:
                icon = "✅"
                color = "green"
            elif node == current:
                icon = "🔄"
                color = "orange"
            else:
                icon = "⬜"
                color = "gray"
            st.markdown(
                f"<div style='text-align:center;font-size:18px'>{icon}</div>"
                f"<div style='text-align:center;font-size:11px;color:{color}'>{label}</div>",
                unsafe_allow_html=True,
            )

    # 当前步骤提示
    if summary["current_label"]:
        st.info(f"当前步骤：**{summary['current_label']}**")

    # 最近的 error/install 信息
    recent_errors = [r for r in records[-10:] if r.get("status") == "error"]
    if recent_errors:
        with st.expander("⚠️ 错误信息"):
            for r in recent_errors:
                st.code(f"[{r['ts']}] {r['label']}: {r.get('msg', '')}", language="text")

    # 安装日志（从 exec_log 末尾读取）
    exec_log = os.path.join(module_folder, "exec.log")
    if os.path.exists(exec_log):
        with open(exec_log, "r", encoding="utf-8") as f:
            log_content = f.read()
        if "自动安装" in log_content or "已安装" in log_content:
            lines = [l for l in log_content.splitlines() if "安装" in l]
            if lines:
                with st.expander("📦 自动安装记录"):
                    st.code("\n".join(lines[-10:]), language="text")


def _read_bg_status(module_folder: str) -> dict:
    """读取后台线程状态文件，返回 {status, data}。"""
    status_file = os.path.join(module_folder, "bg_status.json")
    if not os.path.exists(status_file):
        return {"status": "starting", "data": {}}
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"status": "starting", "data": {}}


# ── 自动实验 Tab ──────────────────────────────────────────────────────────────
with tab_auto:
    from yanzhi.llm import models

    # ── 初始化 session state ─────────────────────────────────────────────────
    defaults = {
        "exp_status":       "idle",
        "exp_module_folder": None,
        "exp_data":         None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    status = st.session_state["exp_status"]
    module_folder = st.session_state.get("exp_module_folder")

    # ── idle — 配置与启动 ─────────────────────────────────────────────────────
    if status == "idle":
        st.markdown("系统将根据「研究方法」自动生成代码、执行、查文献并迭代优化，流程中会请你确认关键决策。")

        col1, col2, col3 = st.columns(3)
        with col1:
            model_choice = st.selectbox(
                "LLM 模型", options=list(models.keys()),
                index=list(models.keys()).index("gemini-2.0-flash") if "gemini-2.0-flash" in models else 0,
                key="exp_model",
            )
        with col2:
            max_outer = st.slider("最大外层优化轮数", 1, 3, 2, key="exp_max_outer")
        with col3:
            timeout = st.number_input("代码执行超时（秒）", 30, 600, 120, 30, key="exp_timeout")

        methods_path = os.path.join(yz.project_dir, "input_files", "methods.md")
        if not os.path.exists(methods_path):
            st.warning("⚠️ 请先在「研究方法」页面生成或上传 methods.md。")
        else:
            if st.button("▶️ 开始自动实验", type="primary", key="run_exp_btn"):
                try:
                    folder = yz.run_experiment_bg_start(
                        llm=model_choice,
                        max_outer_iter=max_outer,
                        timeout=timeout,
                    )
                    st.session_state["exp_module_folder"] = folder
                    st.session_state["exp_status"] = "running_bg"
                except Exception as e:
                    st.session_state["exp_status"] = "error"
                    st.session_state["exp_data"] = str(e)
                st.rerun()

    # ── running_bg — 轮询后台线程状态 ────────────────────────────────────────
    elif status == "running_bg":
        st.markdown("### 🔬 实验进行中")

        if module_folder and os.path.exists(module_folder):
            _render_step_progress(module_folder)

        bg = _read_bg_status(module_folder or "")
        bg_status = bg["status"]
        bg_data   = bg["data"]

        if bg_status == "starting":
            # 线程刚启动，稍等
            time.sleep(1)
            st.rerun()
        elif bg_status == "running":
            time.sleep(2)
            st.rerun()
        elif bg_status in ("waiting_dataset", "waiting_criteria", "waiting_lit_review", "waiting_unknown"):
            st.session_state["exp_status"] = bg_status
            st.session_state["exp_data"]   = bg_data
            st.rerun()
        elif bg_status == "done":
            st.session_state["exp_status"] = "done"
            st.session_state["exp_data"]   = bg_data.get("summary", "")
            st.rerun()
        elif bg_status == "error":
            st.session_state["exp_status"] = "error"
            st.session_state["exp_data"]   = bg_data.get("message", "未知错误")
            st.rerun()
        else:
            time.sleep(2)
            st.rerun()

    # ── waiting_dataset — 介入点③：选择数据集 ───────────────────────────────────
    elif status == "waiting_dataset":
        data = st.session_state["exp_data"]
        if isinstance(data, dict):
            st.info(data.get("message", "请选择数据集。"))
            st.markdown(f"**研究领域：** {data.get('domain', '')}　**数据需求：** {data.get('reason', '')}")
            st.markdown(f"**搜索词：** `{data.get('search_query', '')}`")
        else:
            st.info("请选择数据集。")

        candidates = data.get("candidates", []) if isinstance(data, dict) else []

        if candidates:
            # 构建展示选项
            options = {
                f"{i}: [{c['source'].upper()}] {c['name']} — {c['description'][:60]} {c['size_hint']}": i
                for i, c in enumerate(candidates)
            }
            selected_label = st.radio(
                "候选数据集",
                options=list(options.keys()),
                key="dataset_radio",
            )
            selected_idx = options[selected_label]

            # 显示选中数据集的标签
            sel = candidates[selected_idx]
            if sel.get("tags"):
                st.caption("标签：" + " · ".join(sel["tags"]))

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ 确认选择", type="primary", key="dataset_confirm_btn"):
                chosen = candidates[selected_idx] if candidates else {"source": "synthetic", "name": "synthetic", "download_key": "synthetic", "description": "合成数据", "size_hint": "", "tags": []}
                if module_folder:
                    yz.run_experiment_bg_resume(module_folder, chosen)
                st.session_state["exp_status"] = "running_bg"
                st.rerun()
        with col_b:
            if st.button("🔢 使用合成数据", key="dataset_synthetic_btn"):
                synthetic = {"source": "synthetic", "name": "synthetic",
                             "download_key": "synthetic", "description": "合成数据",
                             "size_hint": "", "tags": []}
                if module_folder:
                    yz.run_experiment_bg_resume(module_folder, synthetic)
                st.session_state["exp_status"] = "running_bg"
                st.rerun()

    # ── waiting_criteria — 介入点①：确认验收标准 ─────────────────────────────
    elif status == "waiting_criteria":
        data = st.session_state["exp_data"]

        # 显示已完成步骤
        if module_folder:
            _render_step_progress(module_folder)

        st.divider()
        st.info(data.get("message", "请确认验收标准。") if isinstance(data, dict) else "请确认验收标准。")

        criteria = st.text_area(
            "验收标准（可直接修改）",
            value=data.get("criteria", "") if isinstance(data, dict) else "",
            height=160,
            key="criteria_input",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ 确认并继续", type="primary", key="criteria_confirm_btn"):
                if module_folder:
                    yz.run_experiment_bg_resume(module_folder, criteria)
                st.session_state["exp_status"] = "running_bg"
                st.rerun()
        with col_b:
            if st.button("⛔ 取消", key="criteria_cancel_btn"):
                st.session_state["exp_status"] = "idle"
                st.session_state["exp_module_folder"] = None
                st.rerun()

    # ── waiting_lit_review — 介入点②：文献评审与决策 ─────────────────────────
    elif status == "waiting_lit_review":
        data  = st.session_state["exp_data"]
        outer = data.get("outer", 1) if isinstance(data, dict) else 1

        # 显示已完成步骤
        if module_folder:
            _render_step_progress(module_folder)

        st.divider()
        st.warning(data.get("message", f"第 {outer} 轮实验结果不足。") if isinstance(data, dict) else f"第 {outer} 轮实验结果不足。")

        if isinstance(data, dict):
            st.markdown(f"**诊断问题：** {data.get('diagnosis', '')}")
            st.markdown(f"**文献检索词：** `{data.get('search_query', '')}`")
            with st.expander("📚 找到的相关文献", expanded=True):
                st.text(data.get("literature", "（无）"))

        decision = st.radio(
            "请决定下一步",
            options=["continue", "modify", "stop"],
            format_func=lambda x: {
                "continue": "✅ 按找到的文献继续优化",
                "modify":   "✏️ 我来指定改进方向",
                "stop":     "⛔ 保存当前结果并停止",
            }[x],
            key="lit_decision",
        )

        custom_input = ""
        if decision == "modify":
            custom_input = st.text_area(
                "请描述改进方向（将合并进方法优化提示词）",
                key="lit_custom_input",
            )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("确认", type="primary", key="lit_confirm_btn"):
                if module_folder:
                    yz.run_experiment_bg_resume(
                        module_folder,
                        {"decision": decision, "input": custom_input},
                    )
                st.session_state["exp_status"] = "running_bg"
                st.rerun()
        with col_b:
            if st.button("⛔ 强制停止", key="lit_stop_btn"):
                if module_folder:
                    yz.run_experiment_bg_resume(
                        module_folder,
                        {"decision": "stop", "input": ""},
                    )
                st.session_state["exp_status"] = "running_bg"
                st.rerun()

    # ── done ──────────────────────────────────────────────────────────────────
    elif status == "done":
        st.success("✅ 实验完成！结果已写入 results.md")

        if module_folder:
            _render_step_progress(module_folder)
            st.divider()

        summary = st.session_state.get("exp_data", "")
        if summary:
            st.markdown("**结果摘要：**")
            st.markdown(summary)

        if module_folder:
            exec_log = os.path.join(module_folder, "exec.log")
            if os.path.exists(exec_log):
                with st.expander("查看执行日志"):
                    with open(exec_log, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="text")

        if st.button("🔄 重新开始实验", key="exp_restart_btn"):
            for k in ("exp_status", "exp_module_folder", "exp_data"):
                st.session_state.pop(k, None)
            st.rerun()

    # ── error ─────────────────────────────────────────────────────────────────
    elif status == "error":
        st.error(f"❌ 实验失败：{st.session_state.get('exp_data', '未知错误')}")

        if module_folder:
            _render_step_progress(module_folder)

        if st.button("🔄 重试", key="exp_retry_btn"):
            st.session_state["exp_status"] = "idle"
            st.rerun()


# ── 手动上传 Tab ──────────────────────────────────────────────────────────────
with tab_manual:
    st.markdown("上传本地实验产生的结果文件和图表，供论文生成阶段使用。")
    st.info("💡 请在本地按「研究方法」中的方案运行实验，再将结果上传至此。")

    uploaded_files = st.file_uploader(
        "上传实验结果（Markdown 摘要 + 图表 PNG/JPG）",
        accept_multiple_files=True,
        key="upload_results",
    )
    if uploaded_files:
        plots_dir = os.path.join(yz.project_dir, "input_files", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for f in uploaded_files:
            if f.name.endswith(".md"):
                yz.set_results(f.read().decode("utf-8"))
            else:
                with open(os.path.join(plots_dir, f.name), "wb") as fp:
                    fp.write(f.getbuffer())
        st.success(f"✅ 已上传 {len(uploaded_files)} 个文件")
        st.rerun()

st.divider()

# ── 图表与结果展示（两个 Tab 共用）─────────────────────────────────────────────
plots_dir = os.path.join(yz.project_dir, "input_files", "plots")
plots = [p for p in Path(plots_dir).glob("*") if p.is_file()] if os.path.exists(plots_dir) else []
if plots:
    st.subheader(f"实验图表（{len(plots)} 张）")
    cols = st.columns(min(len(plots), 3))
    for i, plot in enumerate(plots):
        with cols[i % 3]:
            st.image(str(plot), caption=plot.name, use_container_width=True)
    zip_data = create_zip_in_memory(plots_dir)
    st.download_button("⬇️ 下载所有图表", data=zip_data, file_name="plots.zip",
                       mime="application/zip")
else:
    st.caption("尚未生成或上传图表。")

st.divider()
st.subheader("实验结果摘要")
try:
    show_markdown_file(os.path.join(yz.project_dir, "input_files", "results.md"), label="实验结果")
except FileNotFoundError:
    st.caption("尚未生成或上传结果摘要。")
