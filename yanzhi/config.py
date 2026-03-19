from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

DEFAUL_PROJECT_NAME = "项目"

# ── 目录名 ────────────────────────────────────────
INPUT_FILES  = "input_files"
PLOTS_FOLDER = "plots"
PAPER_FOLDER = "paper"
HISTORY_DIR  = "history"

# ── 输入文件名 ────────────────────────────────────
DESCRIPTION_FILE = "data_description.md"
IDEA_FILE        = "idea.md"
METHOD_FILE      = "methods.md"
RESULTS_FILE     = "results.md"
LITERATURE_FILE  = "literature.md"
REFEREE_FILE     = "referee.md"

# ── 论文版本文件名 ────────────────────────────────
PAPER_V1 = "paper_v1_preliminary.tex"
PAPER_V2 = "paper_v2_no_citations.tex"
PAPER_V3 = "paper_v3_citations.tex"
PAPER_V4 = "paper_v4_final.tex"

# ── 重试 & 迭代上限 ──────────────────────────────
JSON_PARSE_MAX_RETRIES   = 5    # JSON 解析最大重试次数
SS_API_MAX_RETRIES       = 10   # Semantic Scholar 最大重试次数
SS_API_RETRY_DELAY       = 1.0  # 重试间隔（秒）
LATEX_FIX_MAX_RETRIES    = 3    # LaTeX 自动修复最大轮次
DEFAULT_IDEA_ITERATIONS  = 4    # get_idea 默认迭代轮数
DEFAULT_LIT_MAX_ITER     = 7    # check_idea 默认最大检索轮数

# ── 历史记录 ──────────────────────────────────────
HISTORY_PREVIEW_CHARS = 200  # 历史版本预览截取字符数

# ── 自动实验 ──────────────────────────────────────────────────────────────
EXPERIMENT_DIR          = "experiment_output"   # 实验输出目录名
DEFAULT_EXEC_TIMEOUT    = 120                   # 默认代码执行超时（秒）
DEFAULT_EXP_MAX_ITER    = 3                     # 默认最大优化轮数
