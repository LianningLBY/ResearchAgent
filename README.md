# 研智 (YanZhi)

**中文多智能体科研助手** — 基于 LangGraph 构建的全流程科研辅助系统。

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 功能

- 🎯 **研究方向** — 支持从"只有一个方向"到"已有数据集"两种起点，AI 辅助撰写
- 📚 **文献调研** — 基于 Semantic Scholar API 检索相关文献，验证研究新颖性
- 💡 **研究想法** — IdeaMaker/IdeaHater 对抗迭代生成创新想法，支持多轮对话修改
- 🔬 **研究方法** — 自动生成实验方案和技术路线，支持多轮对话修改
- 🤖 **自动实验** — 根据研究方法全自动运行实验，含：
  - 自动发现并下载公开数据集（sklearn / HuggingFace / Papers With Code）
  - 自动生成、执行、修复实验代码，缺包自动安装
  - Human-in-the-Loop：关键节点暂停请用户确认，支持文献辅助迭代优化
  - 实时进度可视化（步骤时间轴，后台线程不阻塞 UI）
- 📝 **论文生成** — 生成中文学术论文（LaTeX 格式，ctex 中文排版，支持知网/IEEE格式）
- ✅ **论文审阅** — AI 同行评审，输出详细审稿意见

**工程特性：**
- 多项目管理：每个课题独立项目，互不干扰
- 版本历史：每次生成自动备份，随时加载旧版本
- 对话修改：生成后直接聊天调整，满意再保存
- 纯 LangGraph 后端，无需 cmbagent

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/LianningLBY/ResearchAgent.git
cd ResearchAgent
```

### 2. 安装依赖

```bash
pip install streamlit langchain-google-genai langchain-anthropic langchain-openai \
            langgraph pymupdf jsonschema json5 requests
```

> 也可以用 editable 模式一次安装所有依赖：`pip install -e ".[app]"`

### 3. 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env，填入至少一个 LLM 的 API Key（推荐 Gemini）
```

### 4. 启动

```bash
streamlit run yanzhi_app/app.py
```

浏览器打开 `http://localhost:8501`

---

## Python 调用

```python
from yanzhi import YanZhi, Journal

yz = YanZhi(project_dir="我的项目")

# 设置研究方向
yz.set_data_description("探索 Transformer 在小样本医学图像分类中的应用")

# 生成研究想法（IdeaMaker/IdeaHater 对抗迭代）
yz.get_idea(llm="gemini-2.0-flash", iterations=4)

# 文献查新
yz.check_idea()

# 生成研究方法
yz.get_method(llm="gemini-2.0-flash")

# 自动实验（Human-in-the-Loop，含数据集发现与下载）
result = yz.run_experiment_start(llm="gemini-2.0-flash", thread_id="exp-1")
# result["status"] == "waiting_dataset" / "waiting_criteria" / "waiting_lit_review" / "done"

# 生成论文（ctexart 格式）
yz.get_paper(journal=Journal.NONE)

# 对话式修改
revised = yz.chat_revise(
    content_type="研究想法",
    current_content=yz.research.idea,
    user_message="请聚焦在小样本场景，去掉全量数据的部分",
)
```

---

## 配置

参见 [`.env.example`](.env.example)，包含：
- LLM API 密钥（Gemini / OpenAI / Anthropic / Perplexity）
- 云数据库配置（Supabase，可选）

---

## 项目结构

```
yanzhi/                        # 后端：LangGraph 多智能体，中文提示词，ctex LaTeX
  experiment_agents/           # 自动实验模块
    dataset_finder.py          # 数据集搜索（sklearn / HuggingFace / Papers With Code）
    data_fetch.py              # 数据集下载与 schema 描述
    env_setup.py               # 缺包检测与动态安装（ensurepip 自举）
    executor.py                # 代码执行（自动安装缺失包，auto-save 图表）
    progress.py                # 实验进度日志（progress.jsonl）
    experiment_node.py         # LangGraph 节点（含3个 HITL 介入点）
    agents_graph.py            # 实验工作流图
yanzhi_app/                    # 前端：Streamlit 多页面仪表盘
  app.py                       # 入口，自动注入 sys.path，多项目管理
  pages/                       # 各功能页面
  utils.py                     # 共享工具（对话修改、历史版本等）
.env.example                   # 环境变量配置模板
```
