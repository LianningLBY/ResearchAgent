from langchain_core.runnables import RunnableConfig
from .parameters import GraphState
from .prompts import novelty_prompt, summary_literature_prompt
from ..paper_agents.tools import extract_latex_block, LLM_call_stream, json_parser3
from ..config import JSON_PARSE_MAX_RETRIES, SS_API_MAX_RETRIES, SS_API_RETRY_DELAY
from ..log import get_logger
import time
import requests

logger = get_logger(__name__)


def novelty_decider(state: GraphState, config: RunnableConfig):
    logger.info("正在判断想法新颖性：第 %d 轮", state['literature']['iteration'])

    PROMPT = novelty_prompt(state)

    for attempt in range(JSON_PARSE_MAX_RETRIES):
        state, result = LLM_call_stream(PROMPT, state)
        try:
            result   = json_parser3(result)
            reason   = result["Reason"]
            decision = result["Decision"]
            query    = result["Query"]
            messages = (f"{state['literature']['messages']}\n"
                        f"第 {state['literature']['iteration']} 轮\n"
                        f"决策：{decision}\n理由：{reason}\n")
            iteration = state['literature']['iteration'] + 1
            break
        except Exception as e:
            logger.warning("JSON 解析失败（第 %d/%d 次）：%s\nLLM 原始输出（前300字）：%s",
                           attempt + 1, JSON_PARSE_MAX_RETRIES, e,
                           str(result)[:300] if result else "(空)")
            time.sleep(2)
    else:
        raise RuntimeError(f"尝试 {JSON_PARSE_MAX_RETRIES} 次后仍无法解析新颖性判断 JSON")

    if 'not novel' in decision.lower():
        logger.info("决策：不新颖")
        return {"literature": {**state['literature'], "reason": reason, "messages": messages,
                               "decision": decision, "query": query, "iteration": iteration,
                               'next_agent': "literature_summary"}}

    elif 'novel' in decision.lower() or iteration >= state['literature']['max_iterations']:
        decision = 'novel'
        logger.info("决策：新颖")
        return {"literature": {**state['literature'], "reason": reason, "messages": messages,
                               "decision": decision, "query": query, "iteration": iteration,
                               'next_agent': "literature_summary"}}
    else:
        logger.info("决策：继续查询，查询语句：%s", query)
        return {"literature": {**state['literature'], "reason": reason, "messages": messages,
                               "decision": decision, "query": query, "iteration": iteration,
                               'next_agent': "semantic_scholar"}}


def semantic_scholar(state: GraphState, config: RunnableConfig):
    results = _ss_api(state['literature']['query'], limit=10)

    total_papers = results.get("total", 0)
    papers       = results.get("data", [])

    papers_str = []
    if papers:
        logger.info("找到 %d 篇潜在相关论文", total_papers)
        for idx, paper in enumerate(papers, start=0):
            authors  = ", ".join([a.get("name", "未知") for a in paper.get("authors", [])])
            title    = paper.get("title",    "无标题")
            year     = paper.get("year",     "未知年份")
            abstract = paper.get("abstract", "无摘要")
            url      = paper.get("url",      "无链接")
            paper_str = f"{idx+state['literature']['num_papers']}. {title} ({year})\n作者：{authors}\n摘要：{abstract}\nURL：{url}\n\n"
            with open(state['files']['literature_log'], 'a', encoding='utf-8') as f:
                f.write(paper_str)
            with open(state['files']['papers'], 'a', encoding='utf-8') as f:
                f.write(paper_str)
            papers_str.append(f"{idx+state['literature']['num_papers']}. {title}\n摘要：{abstract}\nURL：{url}\n")
    else:
        papers_str.append("未找到相关论文。\n")

    total_found = state['literature']['num_papers'] + min(len(papers), 10)
    logger.info("已分析论文总数：%d", total_found)
    return {"literature": {**state['literature'], 'papers': papers_str, "num_papers": total_found}}


def _ss_api(query: str, limit: int = 10) -> dict:
    """调用 Semantic Scholar API，失败时按指数退避重试。"""
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,authors,year,abstract,url"}
    delay = SS_API_RETRY_DELAY
    for attempt in range(SS_API_MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                logger.warning("Semantic Scholar 限流，等待 %.1f 秒后重试（%d/%d）…",
                               delay, attempt + 1, SS_API_MAX_RETRIES)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            logger.error("Semantic Scholar 请求失败：状态码 %d，内容：%s",
                         response.status_code, response.text[:200])
            return {}
        except requests.RequestException as e:
            logger.warning("Semantic Scholar 网络异常（%d/%d）：%s，%.1f 秒后重试…",
                           attempt + 1, SS_API_MAX_RETRIES, e, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    logger.error("Semantic Scholar 达到最大重试次数（%d），返回空结果。", SS_API_MAX_RETRIES)
    return {}


def literature_summary(state: GraphState, config: RunnableConfig):
    PROMPT = summary_literature_prompt(state)
    state, result = LLM_call_stream(PROMPT, state)

    if not result or not result.strip():
        logger.error("literature_summary：LLM 返回空内容，跳过报告写入。")
        return {}

    text = result.strip()

    with open(state['files']['literature'], 'w', encoding='utf-8') as f:
        f.write(f"想法{'新颖' if state['literature']['decision']=='novel' else '不新颖'}\n\n")
        f.write(text)

    logger.info("文献综述完成。累计 token：输入 %d，输出 %d", state['tokens']['ti'], state['tokens']['to'])
    print(text)
