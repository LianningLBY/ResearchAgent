"""数据集下载与 Schema 描述。

支持：sklearn 内置 / HuggingFace datasets / 直链 URL / 合成数据（兜底）
下载结果缓存到 {project_dir}/datasets/，重跑不重下。
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
import tarfile
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset_finder import DatasetCandidate

from ..log import get_logger

logger = get_logger(__name__)

_FETCH_TIMEOUT = 300   # 下载最长等待秒数


# ── 主接口 ────────────────────────────────────────────────────────────────────

def fetch_dataset(candidate: "DatasetCandidate", cache_dir: str,
                  hf_token: str | None = None) -> dict:
    """
    下载/加载数据集到 cache_dir，返回：
    {
      success: bool,
      local_path: str | None,     # 主数据文件或目录
      schema_md: str,             # 供 LLM 读取的 Markdown schema 描述
      error: str,                 # 失败原因（success=False 时）
    }
    """
    os.makedirs(cache_dir, exist_ok=True)
    src = candidate.source

    if src == "synthetic":
        return {"success": True, "local_path": None,
                "schema_md": "（使用合成数据，由实验代码自动生成）", "error": ""}

    if src == "sklearn":
        return _fetch_sklearn(candidate, cache_dir)

    if src == "huggingface":
        return _fetch_huggingface(candidate, cache_dir, hf_token=hf_token)

    if src in ("paperswithcode", "url"):
        url = candidate.download_key
        if url.startswith("http"):
            return _fetch_url(url, candidate.name, cache_dir)
        return {"success": False, "local_path": None,
                "schema_md": "", "error": f"无可用下载链接：{url}"}

    return {"success": False, "local_path": None,
            "schema_md": "", "error": f"未知数据源：{src}"}


# ── sklearn ───────────────────────────────────────────────────────────────────

def _fetch_sklearn(candidate: "DatasetCandidate", cache_dir: str) -> dict:
    local_path = os.path.join(cache_dir, f"{candidate.name}.csv")
    try:
        # 自动安装缺失依赖
        try:
            import sklearn.datasets  # noqa: F401
        except ImportError:
            logger.info("自动安装 scikit-learn...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "scikit-learn", "-q"],
                capture_output=True, timeout=180,
            )
        try:
            import pandas  # noqa: F401
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pandas", "-q"],
                capture_output=True, timeout=120,
            )

        if not os.path.exists(local_path):
            import sklearn.datasets as skd
            import pandas as pd
            import numpy as np

            loader = getattr(skd, candidate.download_key)
            bunch  = loader()

            # 处理结构化数组（如 kddcup99）和普通数组两种情况
            data = bunch.data
            if hasattr(data, "dtype") and data.dtype.names:
                # 结构化 numpy 数组 → 直接转 DataFrame
                df = pd.DataFrame(data)
            else:
                feature_names = (
                    list(bunch.feature_names)
                    if hasattr(bunch, "feature_names")
                    else [f"feature_{i}" for i in range(data.shape[1])]
                )
                df = pd.DataFrame(data, columns=feature_names)

            if hasattr(bunch, "target"):
                target = bunch.target
                target_names = getattr(bunch, "target_names", None)
                # target 可能是 bytes 数组（kddcup99），直接解码；也可能是整数索引
                if len(target) > 0 and isinstance(target[0], (bytes, np.bytes_)):
                    df["target"] = [t.decode("utf-8", errors="replace") for t in target]
                elif target_names is not None:
                    try:
                        df["target"] = [target_names[t] for t in target]
                    except (IndexError, TypeError):
                        df["target"] = target
                else:
                    df["target"] = target
            df.to_csv(local_path, index=False)
            logger.info("sklearn 数据集已保存：%s（%d 行）", local_path, len(df))

        schema_md = _describe_file(local_path)
        return {"success": True, "local_path": local_path, "schema_md": schema_md, "error": ""}
    except Exception as e:
        logger.error("sklearn 加载失败：%s", e)
        return {"success": False, "local_path": None, "schema_md": "", "error": str(e)}


# ── HuggingFace ───────────────────────────────────────────────────────────────

def _fetch_huggingface(candidate: "DatasetCandidate", cache_dir: str,
                       hf_token: str | None = None) -> dict:
    safe_name  = candidate.download_key.replace("/", "_")
    local_path = os.path.join(cache_dir, f"{safe_name}.parquet")
    access_type = getattr(candidate, "access_type", "public")
    try:
        if not os.path.exists(local_path):
            _ensure_hf_datasets()
            from datasets import load_dataset
            # gated 数据集需要 token；无 token 则提前返回友好提示
            if access_type == "gated" and not hf_token:
                return {
                    "success": False, "local_path": None, "schema_md": "",
                    "error": "gated_no_token",
                    "access_note": getattr(candidate, "access_note", ""),
                }
            if access_type == "private":
                return {
                    "success": False, "local_path": None, "schema_md": "",
                    "error": "private_dataset",
                    "access_note": getattr(candidate, "access_note", ""),
                }
            logger.info("正在从 HuggingFace 下载：%s", candidate.download_key)
            ds = load_dataset(
                candidate.download_key, split="train",
                trust_remote_code=False,
                token=hf_token or None,
            )
            ds.to_parquet(local_path)
            logger.info("HuggingFace 数据集已保存：%s（%d 行）", local_path, len(ds))

        schema_md = _describe_file(local_path)
        return {"success": True, "local_path": local_path, "schema_md": schema_md, "error": ""}
    except Exception as e:
        err_str = str(e)
        if "gated" in err_str.lower() or "access" in err_str.lower():
            return {
                "success": False, "local_path": None, "schema_md": "",
                "error": "gated_no_token",
                "access_note": getattr(candidate, "access_note", ""),
            }
        logger.error("HuggingFace 下载失败：%s", e)
        return {"success": False, "local_path": None, "schema_md": "", "error": err_str}


def _ensure_hf_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError:
        logger.info("安装 huggingface datasets 库...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "datasets", "-q"],
            capture_output=True, timeout=120,
        )


# ── URL 直链 ──────────────────────────────────────────────────────────────────

def _fetch_url(url: str, name: str, cache_dir: str) -> dict:
    import urllib.request
    safe_name = name.replace("/", "_").replace(" ", "_")
    suffix    = Path(url.split("?")[0]).suffix or ".bin"
    raw_path  = os.path.join(cache_dir, f"{safe_name}{suffix}")

    try:
        if not os.path.exists(raw_path):
            logger.info("正在下载：%s", url)
            urllib.request.urlretrieve(url, raw_path)
            logger.info("下载完成：%s", raw_path)

        # 解压 zip / tar
        extracted = _try_extract(raw_path, cache_dir)
        final_path = extracted or raw_path

        schema_md = _describe_file(final_path)
        return {"success": True, "local_path": final_path, "schema_md": schema_md, "error": ""}
    except Exception as e:
        logger.error("URL 下载失败：%s", e)
        return {"success": False, "local_path": None, "schema_md": "", "error": str(e)}


def _try_extract(path: str, dest: str) -> str | None:
    """尝试解压，返回解压后的主数据文件路径，失败返回 None。"""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        # 找解压出的数据文件
        return _find_data_file(dest)
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(dest)
        return _find_data_file(dest)
    return None


def _find_data_file(directory: str) -> str | None:
    """在目录中找最大的 csv/parquet/json/tsv 文件。"""
    candidates = []
    for ext in ("*.csv", "*.parquet", "*.json", "*.tsv", "*.jsonl"):
        candidates.extend(Path(directory).rglob(ext))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_size))


# ── Schema 描述 ───────────────────────────────────────────────────────────────

def _describe_file(path: str) -> str:
    """读取数据文件，生成供 LLM 阅读的 Markdown schema 描述。"""
    try:
        import pandas as pd
        p = Path(path)

        if p.suffix == ".csv":
            df = pd.read_csv(path, nrows=5)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(path).head(5)
        elif p.suffix in (".json", ".jsonl"):
            df = pd.read_json(path, lines=p.suffix == ".jsonl", nrows=5)
        elif p.suffix == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=5)
        else:
            return f"文件路径：{path}（格式未知，由代码自行读取）"

        lines = [
            f"**本地路径：** `{path}`",
            f"**行数：** ~{_estimate_rows(path, df)}  **列数：** {len(df.columns)}",
            "",
            "**列名与类型：**",
        ]
        for col, dtype in df.dtypes.items():
            lines.append(f"- `{col}` ({dtype})")
        lines += ["", "**前 3 行样本：**", "```", df.head(3).to_string(index=False), "```"]
        return "\n".join(lines)
    except Exception as e:
        return f"文件路径：{path}（读取 schema 失败：{e}）"


def _estimate_rows(path: str, sample_df) -> str:
    try:
        size  = os.path.getsize(path)
        if len(sample_df) == 0:
            return "未知"
        row_size = size / max(len(sample_df), 1)
        return f"{int(size / row_size):,}"
    except Exception:
        return "未知"
