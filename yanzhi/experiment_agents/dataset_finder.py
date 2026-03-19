"""数据集发现：根据研究方法自动搜索合适的公开数据集。

支持来源：
  - sklearn 内置数据集（零配置）
  - HuggingFace Hub（免认证 API）
  - Papers With Code（免认证 API）
  - 合成数据（始终可用，作为兜底）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from ..log import get_logger
from ..config import SS_API_RETRY_DELAY, SS_API_MAX_RETRIES

logger = get_logger(__name__)

_HF_API        = "https://huggingface.co/api/datasets"
_PWC_API       = "https://paperswithcode.com/api/v1/datasets/"
_REQUEST_TIMEOUT = 10


@dataclass
class DatasetCandidate:
    name:          str
    source:        str          # "sklearn" | "huggingface" | "paperswithcode" | "synthetic"
    description:   str
    download_key:  str          # sklearn: "load_iris"; HF: "owner/repo"; synthetic: ""
    size_hint:     str          # 大致体量描述
    tags:          list[str] = field(default_factory=list)
    access_type:   str = "public"   # "public" | "gated" | "private"
    access_note:   str = ""         # 获取方式说明（gated/private 时填写）

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "source":       self.source,
            "description":  self.description,
            "download_key": self.download_key,
            "size_hint":    self.size_hint,
            "tags":         self.tags,
            "access_type":  self.access_type,
            "access_note":  self.access_note,
        }

    @staticmethod
    def from_dict(d: dict) -> "DatasetCandidate":
        return DatasetCandidate(
            name=d["name"], source=d["source"], description=d["description"],
            download_key=d["download_key"], size_hint=d["size_hint"],
            tags=d.get("tags", []),
            access_type=d.get("access_type", "public"),
            access_note=d.get("access_note", ""),
        )


# ── sklearn 内置数据集清单 ────────────────────────────────────────────────────

_SKLEARN_CATALOG: list[DatasetCandidate] = [
    DatasetCandidate("iris",               "sklearn", "鸢尾花三分类（150×4）",               "load_iris",               "~5 KB",   ["classification", "multiclass"]),
    DatasetCandidate("breast_cancer",      "sklearn", "乳腺癌二分类（569×30）",               "load_breast_cancer",      "~30 KB",  ["classification", "binary"]),
    DatasetCandidate("digits",             "sklearn", "手写数字识别 0-9（1797×64）",          "load_digits",             "~500 KB", ["classification", "image"]),
    DatasetCandidate("wine",               "sklearn", "葡萄酒品种分类（178×13）",              "load_wine",               "~30 KB",  ["classification"]),
    DatasetCandidate("diabetes",           "sklearn", "糖尿病回归（442×10）",                 "load_diabetes",           "~20 KB",  ["regression"]),
    DatasetCandidate("california_housing", "sklearn", "加州房价回归（20640×8）",              "fetch_california_housing","~1 MB",   ["regression"]),
    DatasetCandidate("linnerud",           "sklearn", "体能指标多输出回归（20×3/3）",          "load_linnerud",           "~5 KB",   ["regression", "multioutput"]),
    DatasetCandidate("20newsgroups",       "sklearn", "20类新闻文本分类（~18k 文档）",         "fetch_20newsgroups",      "~14 MB",  ["classification", "nlp", "text"]),
    DatasetCandidate("olivetti_faces",     "sklearn", "Olivetti 人脸识别（400×4096）",        "fetch_olivetti_faces",    "~1.5 MB", ["classification", "image"]),
    DatasetCandidate("covtype",            "sklearn", "森林植被类型分类（581012×54）",         "fetch_covtype",           "~75 MB",  ["classification", "tabular"]),
    DatasetCandidate("kddcup99",           "sklearn", "KDD Cup 99 网络入侵检测",              "fetch_kddcup99",          "~18 MB",  ["classification", "network", "anomaly"]),
    DatasetCandidate("rcv1",               "sklearn", "Reuters 新闻多标签分类（804414×47236）","fetch_rcv1",              "~656 MB", ["classification", "nlp", "multilabel"]),
]

_SKLEARN_KEYWORD_MAP: dict[str, list[str]] = {
    # 中文领域词
    "分类":     ["classification"],
    "回归":     ["regression"],
    "文本":     ["nlp", "text"],
    "图像":     ["image"],
    "网络安全": ["network", "anomaly"],
    "网络流量": ["network", "anomaly"],
    "入侵检测": ["network", "anomaly"],
    "网络":     ["network"],
    "入侵":     ["network", "anomaly"],
    "异常":     ["anomaly"],
    "医疗":     ["binary", "classification"],
    "房价":     ["regression"],
    "花":       ["classification", "multiclass"],
    # 英文领域词
    "classification":    ["classification"],
    "regression":        ["regression"],
    "nlp":               ["nlp"],
    "image":             ["image"],
    "network":           ["network"],
    "anomaly":           ["anomaly"],
    "intrusion":         ["network", "anomaly"],
    "security":          ["network", "anomaly"],
    "traffic":           ["network", "anomaly"],
    "attack":            ["network", "anomaly"],
    "malware":           ["network", "anomaly"],
    "cybersecurity":     ["network", "anomaly"],
    "pcap":              ["network", "anomaly"],
    "tls":               ["network", "anomaly"],
    "encrypted":         ["network", "anomaly"],
    "detection":         ["anomaly", "classification"],
}

# 与 sklearn 强相关的领域词 —— 若查询主要命中这些词才推荐 sklearn
_SKLEARN_MIN_SCORE = 3


def _sklearn_by_query(query: str, top_k: int = 3) -> list[DatasetCandidate]:
    """根据查询词从 sklearn 目录中筛选最相关的数据集。

    只有评分 >= _SKLEARN_MIN_SCORE 的数据集才会被返回，避免不相关推荐。
    """
    q = query.lower()
    scored: list[tuple[int, DatasetCandidate]] = []
    for ds in _SKLEARN_CATALOG:
        score = 0
        text = (ds.name + ds.description + " ".join(ds.tags)).lower()
        # 直接关键词命中（数据集名/描述/标签）
        for word in q.split():
            if len(word) >= 3 and word in text:
                score += 2
        # 领域映射：查询词中的领域关键词 → 标签匹配
        for kw, tag_list in _SKLEARN_KEYWORD_MAP.items():
            if kw in q:
                for tag in tag_list:
                    if tag in ds.tags:
                        score += 2
        # 只收录评分达到阈值的结果，过滤掉凑数的无关数据集
        if score >= _SKLEARN_MIN_SCORE:
            scored.append((score, ds))
    scored.sort(key=lambda x: -x[0])
    return [ds for _, ds in scored[:top_k]]


# ── HuggingFace Hub API ───────────────────────────────────────────────────────

def _search_huggingface(query: str, limit: int = 5) -> list[DatasetCandidate]:
    """搜索 HuggingFace 数据集，若首次无结果则自动用简化词重试。"""
    def _do_search(q: str) -> list[DatasetCandidate]:
        try:
            resp = requests.get(
                _HF_API,
                params={"search": q, "limit": limit, "full": "true"},
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning("HuggingFace API 返回 %d", resp.status_code)
                return []
            results = []
            for d in resp.json():
                did   = d.get("id", "")
                desc  = (d.get("description") or did)[:180]
                tags  = [t for t in d.get("tags", []) if not t.startswith("license:")][:5]
                gated = d.get("gated", False)
                private = d.get("private", False)
                if private:
                    access_type = "private"
                    access_note = "私有数据集，需联系所有者获取访问权限"
                elif gated:
                    access_type = "gated"
                    gate_str = gated if isinstance(gated, str) else "manual"
                    access_note = (
                        f"受限数据集（{gate_str}），需在 HuggingFace 申请后使用 HF Token 下载。"
                        f"申请地址：https://huggingface.co/datasets/{did}"
                    )
                else:
                    access_type = "public"
                    access_note = ""
                results.append(DatasetCandidate(
                    name=did, source="huggingface", description=desc,
                    download_key=did, size_hint="", tags=tags,
                    access_type=access_type, access_note=access_note,
                ))
            return results
        except Exception as e:
            logger.warning("HuggingFace 搜索失败：%s", e)
            return []

    results = _do_search(query)
    if results:
        return results

    # 第一次无结果 → 取前3个英文词重试（更宽泛）
    words = [w for w in query.split() if w.isalpha() and len(w) >= 4]
    if len(words) >= 2:
        short_query = " ".join(words[:3])
        logger.info("HuggingFace 首次搜索无结果，用简化词重试：%s", short_query)
        results = _do_search(short_query)

    return results


# ── Papers With Code API ──────────────────────────────────────────────────────

def _search_paperswithcode(query: str, limit: int = 4) -> list[DatasetCandidate]:
    try:
        resp = requests.get(
            _PWC_API,
            params={"q": query, "page_size": limit},
            timeout=_REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        results = []
        for d in resp.json().get("results", []):
            name = d.get("name", "")
            desc = (d.get("description") or name)[:180]
            tags = list(d.get("modalities") or [])[:4]
            url  = d.get("url", "")
            results.append(DatasetCandidate(
                name=name,
                source="paperswithcode",
                description=desc,
                download_key=url,
                size_hint="",
                tags=tags,
            ))
        return results
    except Exception as e:
        logger.warning("Papers With Code 搜索失败：%s", e)
        return []


# ── 合成数据兜底 ──────────────────────────────────────────────────────────────

_SYNTHETIC = DatasetCandidate(
    name="synthetic",
    source="synthetic",
    description="根据研究方法自动生成合成数据（无需下载，立即可用，适合算法验证）",
    download_key="synthetic",
    size_hint="即时生成",
    tags=["any"],
)


# ── 主接口 ────────────────────────────────────────────────────────────────────

def search_datasets(query: str,
                    hf_limit: int = 5,
                    pwc_limit: int = 3) -> list[DatasetCandidate]:
    """
    根据查询词搜索合适的公开数据集，返回候选列表。
    末尾始终附加「合成数据」选项作为兜底。
    """
    logger.info("搜索数据集：%s", query)
    candidates: list[DatasetCandidate] = []

    sklearn_hits = _sklearn_by_query(query, top_k=2)
    candidates.extend(sklearn_hits)

    hf_hits = _search_huggingface(query, limit=hf_limit)
    candidates.extend(hf_hits)

    pwc_hits = _search_paperswithcode(query, limit=pwc_limit)
    candidates.extend(pwc_hits)

    # 去重（同名只保留第一个）
    seen: set[str] = set()
    unique: list[DatasetCandidate] = []
    for c in candidates:
        key = c.name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    unique.append(_SYNTHETIC)
    logger.info("找到 %d 个候选数据集（含合成数据）", len(unique))
    return unique
