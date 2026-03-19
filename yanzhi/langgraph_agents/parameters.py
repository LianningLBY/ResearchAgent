from typing_extensions import TypedDict, Any
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from ..key_manager import KeyManager


class FILES(TypedDict):
    Folder: str
    data_description: str
    LLM_calls: str
    Temp: str
    idea: str
    methods: str
    idea_log: str
    literature: str
    literature_log: str
    papers: str
    referee_report: str
    referee_log: str
    paper_images: str
    Error: str
    module_folder: str
    f_stream: str


class TOKENS(TypedDict):
    ti: int
    to: int
    i:  int
    o:  int


class LLM(TypedDict):
    model: str
    max_output_tokens: int
    llm: Any
    temperature: float
    stream_verbose: bool


class IDEA(TypedDict):
    iteration: int
    previous_ideas: str
    idea: str
    criticism: str
    total_iterations: int


class REFEREE(TypedDict):
    paper_version: int
    report: str
    images: list[str]


class LITERATURE(TypedDict):
    iteration: int
    query: str
    decision: str
    papers: str
    next_agent: str
    messages: str
    max_iterations: int
    num_papers: int


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    idea: IDEA
    tokens: TOKENS
    llm: LLM
    files: FILES
    keys: KeyManager
    data_description: str
    literature_text: str
    task: str
    literature: LITERATURE
    referee: REFEREE
