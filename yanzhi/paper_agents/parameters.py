from typing_extensions import TypedDict, Any
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .journal import Journal
from ..key_manager import KeyManager


class PAPER(TypedDict):
    Title: str
    Abstract: str
    Keywords: str
    Introduction: str
    Methods: str
    Results: str
    Conclusions: str
    References: str
    summary: str
    journal: Journal
    add_citations: bool


class FILES(TypedDict):
    Folder: str
    Idea: str
    Methods: str
    Results: str
    Plots: str
    Paper_v1: str
    Paper_v2: str
    Paper_v3: str
    Paper_v4: str
    Error: str
    LaTeX_log: str
    LaTeX_err: str
    Temp: str
    LLM_calls: str
    Paper_folder: str
    num_plots: int


class IDEA(TypedDict):
    Idea: str
    Methods: str
    Results: str


class TOKENS(TypedDict):
    ti: int
    to: int
    i: int
    o: int


class LATEX(TypedDict):
    section_to_fix: str


class LLM(TypedDict):
    model: str
    max_output_tokens: int
    llm: Any
    temperature: float


class TIME(TypedDict):
    start: float


class PARAMS(TypedDict):
    num_keywords: int


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    files: FILES
    idea: IDEA
    paper: PAPER
    tokens: TOKENS
    llm: LLM
    latex: LATEX
    keys: KeyManager
    time: TIME
    writer: str
    params: PARAMS
