"""实验环境自动配置：缺包检测 + 动态安装。"""
import re
import sys
import subprocess
from ..log import get_logger

logger = get_logger(__name__)

# 只自举一次 pip，避免重复运行
_pip_bootstrapped = False


def _ensure_pip() -> bool:
    """确保当前 Python 环境有 pip。若缺失则用 ensurepip 自举，返回是否成功。"""
    global _pip_bootstrapped
    if _pip_bootstrapped:
        return True

    # 先检查 pip 是否可用
    check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        _pip_bootstrapped = True
        return True

    # pip 不可用 → 用 ensurepip 自举
    logger.info("当前环境缺少 pip，尝试用 ensurepip 自举...")
    try:
        import ensurepip
        ensurepip.bootstrap(upgrade=True)
        _pip_bootstrapped = True
        logger.info("ensurepip 自举成功")
        return True
    except Exception:
        pass

    # ensurepip 失败 → 用 subprocess 再试一次
    result = subprocess.run(
        [sys.executable, "-m", "ensurepip", "--upgrade"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        _pip_bootstrapped = True
        logger.info("ensurepip subprocess 自举成功")
        return True

    logger.error("无法自举 pip，动态安装将失败。")
    return False

# 模块名 → pip 包名的映射（模块名与包名不同的情况）
MODULE_TO_PACKAGE: dict[str, str] = {
    "sklearn":      "scikit-learn",
    "cv2":          "opencv-python",
    "PIL":          "Pillow",
    "skimage":      "scikit-image",
    "yaml":         "pyyaml",
    "bs4":          "beautifulsoup4",
    "dateutil":     "python-dateutil",
    "attr":         "attrs",
    "pkg_resources": "setuptools",
    "Bio":          "biopython",
    "wx":           "wxPython",
    "gi":           "PyGObject",
}

# 常用科学计算包，一次性预装
STANDARD_SCIENCE_PACKAGES = [
    "numpy", "pandas", "matplotlib", "scipy",
    "scikit-learn", "statsmodels", "seaborn",
]


def extract_missing_module(stderr: str) -> str | None:
    """从 stderr 中提取 ModuleNotFoundError 缺少的模块名。"""
    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", stderr)
    if m:
        return m.group(1).split(".")[0]
    return None


def module_to_package(module_name: str) -> str:
    """将模块名转换为 pip 包名。"""
    return MODULE_TO_PACKAGE.get(module_name, module_name)


def install_package(module_name: str) -> tuple[bool, str]:
    """安装指定模块对应的 pip 包，返回 (成功, 输出信息)。"""
    package = module_to_package(module_name)
    logger.info("自动安装缺失包：%s（模块：%s）", package, module_name)

    # 确保 pip 可用
    if not _ensure_pip():
        return False, "pip 不可用且无法自举"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
        ok  = result.returncode == 0
        out = (result.stdout + result.stderr).strip()
        if ok:
            logger.info("安装成功：%s", package)
        else:
            logger.warning("安装失败：%s\n%s", package, out[:500])
        return ok, out
    except subprocess.TimeoutExpired:
        msg = f"安装超时：{package}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        logger.error("安装异常：%s", e)
        return False, str(e)


def ensure_science_packages() -> list[str]:
    """检测并安装缺失的标准科学计算包，返回安装了的包名列表。"""
    installed = []
    for pkg in STANDARD_SCIENCE_PACKAGES:
        module = pkg.replace("-", "_").replace("scikit_learn", "sklearn")
        try:
            __import__(module)
        except ImportError:
            ok, _ = install_package(module)
            if ok:
                installed.append(pkg)
    return installed
