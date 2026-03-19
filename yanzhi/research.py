from pydantic import BaseModel
from typing import List, Dict


class Research(BaseModel):
    """科研项目数据模型"""
    data_description: str = ""
    idea: str = ""
    methodology: str = ""
    results: str = ""
    plot_paths: List[str] = []
    keywords: Dict[str, str] | list = []
