"""subprocess 代码执行沙箱，支持 matplotlib 自动保存 + 动态包安装。"""
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ..log import get_logger
from .env_setup import extract_missing_module, install_package

logger = get_logger(__name__)

MAX_OUTPUT_CHARS = 10_000
MAX_AUTO_INSTALL = 3   # 最多自动安装 3 个不同的缺失包

_MATPLOTLIB_PREAMBLE = """\
import os as _os
_PLOTS_DIR = _os.environ.get('YZ_PLOTS_DIR', '.')
_plot_counter = [0]

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _auto_save(*args, **kwargs):
        _plot_counter[0] += 1
        _os.makedirs(_PLOTS_DIR, exist_ok=True)
        plt.savefig(_os.path.join(_PLOTS_DIR, f'figure_{_plot_counter[0]}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    plt.show = _auto_save
except ImportError:
    pass
"""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    plot_files: List[str] = field(default_factory=list)
    install_log: str = ""      # 自动安装记录

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def combined_output(self) -> str:
        parts = []
        if self.install_log:
            parts.append(f"[自动安装]\n{self.install_log}")
        if self.stdout.strip():
            parts.append(f"[stdout]\n{self.stdout}")
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr}")
        return "\n".join(parts) if parts else "(无输出)"


def _run_script(code: str, work_dir: str, timeout: int,
                plots_dir: str) -> tuple[str, str, int]:
    """执行一次代码，返回 (stdout, stderr, returncode)。"""
    full_code = _MATPLOTLIB_PREAMBLE + "\n" + code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=work_dir,
        delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(full_code)
        script_path = tmp.name

    env = {**os.environ, "YZ_PLOTS_DIR": plots_dir}
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=work_dir, capture_output=True,
            text=True, timeout=timeout, env=env,
        )
        stdout = proc.stdout[-MAX_OUTPUT_CHARS:] if len(proc.stdout) > MAX_OUTPUT_CHARS else proc.stdout
        stderr = proc.stderr[-MAX_OUTPUT_CHARS:] if len(proc.stderr) > MAX_OUTPUT_CHARS else proc.stderr
        return stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"执行超时（>{timeout}s），进程已终止。", -1
    except Exception as e:
        return "", str(e), -2
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def execute_code(code: str, work_dir: str, timeout: int = 120,
                 plots_dir: str | None = None) -> ExecResult:
    """在 subprocess 中执行 Python 代码，遇到缺包自动安装后重试。

    Args:
        code:      要执行的 Python 代码字符串。
        work_dir:  子进程工作目录。
        timeout:   超时秒数（默认 120s）。
        plots_dir: 图片保存目录；为 None 时使用 work_dir/plots。

    Returns:
        ExecResult 实例（含自动安装日志）。
    """
    if plots_dir is None:
        plots_dir = os.path.join(work_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    installed_modules: set[str] = set()
    install_log_lines: list[str] = []

    stdout, stderr, returncode = _run_script(code, work_dir, timeout, plots_dir)

    # ── 自动安装缺失包并重试 ────────────────────────────────────────────────
    for _ in range(MAX_AUTO_INSTALL):
        if returncode == 0:
            break
        module = extract_missing_module(stderr)
        if not module or module in installed_modules:
            break
        installed_modules.add(module)
        ok, install_out = install_package(module)
        if ok:
            install_log_lines.append(f"✅ 已安装 {module} 对应包")
            logger.info("已安装 %s，重新执行代码...", module)
            stdout, stderr, returncode = _run_script(code, work_dir, timeout, plots_dir)
        else:
            install_log_lines.append(f"❌ 安装 {module} 失败：{install_out[:200]}")
            break

    plot_files = sorted(str(p) for p in Path(plots_dir).glob("figure_*.png"))
    return ExecResult(
        stdout=stdout, stderr=stderr, returncode=returncode,
        plot_files=plot_files,
        install_log="\n".join(install_log_lines),
    )
