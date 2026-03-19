from typing import Callable
from pydantic import BaseModel
from enum import Enum


class Journal(str, Enum):
    """支持的中文期刊格式枚举"""
    NONE  = None
    """标准中文学术格式（ctexart，单栏）"""
    CNKI  = "CNKI"
    """中国知网风格（ctexart，双栏）"""
    IEEE  = "IEEE"
    """IEEE 双栏格式（含 ctex 中文支持）"""


class LatexPresets(BaseModel):
    """各期刊的 LaTeX 预设"""
    article: str
    layout: str = ""
    title: str = r"\title"
    author: Callable[[str], str] = lambda x: f"\\author{{{x}}}"
    bibliographystyle: str = ""
    usepackage: str = ""
    affiliation: Callable[[str], str] = lambda x: rf"\date{{{x}}}"
    abstract: Callable[[str], str]
    files: list[str] = []
    keywords: Callable[[str], str] = lambda x: ""

    class Config:
        arbitrary_types_allowed = True
