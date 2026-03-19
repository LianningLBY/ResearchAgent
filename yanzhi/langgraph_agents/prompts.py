from langchain_core.messages import HumanMessage


def idea_maker_prompt(state):
    literature_block = ""
    if state.get("literature_text"):
        literature_block = f"""
文献调研摘要（如有，优先从中寻找空白/差异化点）：
{state['literature_text']}
"""
    return [HumanMessage(content=rf"""你的目标是为一篇科学论文提出一个突破性的研究想法。请根据以下数据描述，生成一个原创的研究想法。如有其他智能体的批评意见，请充分考虑并改进。务必遵循数据描述中提到的研究方向与约束。

当前迭代轮次：{state['idea']['iteration']}

数据描述：
{state['data_description']}
{literature_block}

历史想法：
{state['idea']['previous_ideas']}

批评意见：
{state['idea']['criticism']}

请严格按照以下格式回复：

\begin{{IDEA}}
<在此填写想法及其简短描述>
\end{{IDEA}}

要求：
- 在 IDEA 块中给出想法标题与 5 句话的描述
- 描述要简洁，不要解释你如何回应了批评意见
""")]


def idea_hater_prompt(state):
    return [HumanMessage(content=rf"""你的目标是对一个研究想法进行严格批评。你将收到这个想法以及生成该想法所基于的数据描述。请从可行性、影响力、创新性等多维度进行严苛批评，目的是帮助改进这个想法。若想法不可行，请建议重新生成。批评时请参考数据描述中的研究方向约束。

数据描述：
{state['data_description']}

历史想法：
{state['idea']['previous_ideas']}

当前想法：
{state['idea']['idea']}

请严格按照以下格式回复：

\begin{{CRITIC}}
<在此填写批评意见>
\end{{CRITIC}}

要求：描述简洁，直击要害。
""")]


def methods_fast_prompt(state):
    return [HumanMessage(content=rf"""你将收到一段数据描述和一个科学论文研究想法。请基于此设计详细的研究方法。

请遵循以下要求：
- 生成详细的方法论描述，清晰列出步骤、技术手段及其依据
- 方法描述须严格聚焦于实现该研究项目所需的流程与手段
- **不得**讨论未来方向、后续工作、扩展或局限性
- 描述风格应如同一位资深研究员向助手解释如何开展研究
- 直接输出方法内容，不要在开头说明你的思考过程

数据描述：
{state['data_description']}

研究想法：
{state['idea']['idea']}

请严格按照以下格式回复：

\begin{{METHODS}}
<在此填写研究方法>
\end{{METHODS}}
""")]


def novelty_prompt(state):
    return [HumanMessage(content=f"""你是一位专业的科学研究助手。请通过与现有文献对比，评估所提研究想法的新颖性。新颖性判断标准如下：

- **不新颖（not novel）**：至少有一篇论文在想法/方法与数据类型上与本想法存在显著重叠。
- **新颖（novel）**：经充分检索后，未发现在想法/方法与数据两方面均存在重大重叠的论文。
- **继续查询（query）**：若未找到相关论文或证据不足，需进一步检索。

重要规则：
- 第一轮的决策必须为 "query"
- 若未返回论文或结果不相关，将 "Decision" 设为 "query"
- 推理中须明确引用最相关论文的标题和 URL
- 推理为单段落（不换行）
- 只输出合法 JSON，不附加任何其他文字
- 若查询过于具体而未找到论文，尝试更宽泛的查询
- 若查询过于宽泛而论文过多，尝试更具体的查询

上下文信息

轮次：{state['literature']['iteration']}/{state['literature']['max_iterations']}

数据描述：{state['data_description']}

研究想法：{state['idea']['idea']}

前几轮消息：{state['literature']['messages']}

本轮找到的论文：
{state['literature']['papers']}
（若迭代轮次为 0，此处无论文）

**请严格按以下格式输出**：
{{
  "Reason": "关于新颖性的单段落推理，引用相关论文标题和 URL。",
  "Decision": "novel | not novel | query",
  "Query": "若 Decision 为 query，给出下一步最优文献检索查询语句（建议用英文以获得更好的 Semantic Scholar 结果）。"
}}
""")]


def summary_literature_prompt(state):
    return [HumanMessage(content=f"""我们有一批数据和一个研究想法，并对文献进行了多轮检索以判断该想法的新颖性。请根据以下信息撰写一份详细的中文总结报告，说明该想法是否具有新颖性。总结中须列出最相似、最相关的论文及其链接，并讨论相似点与不同点。

**数据描述**：
{state['data_description']}

**研究想法**：
{state['idea']['idea']}

**文献检索迭代记录**：
{state['literature']['messages']}

严格要求：
1. 全程使用中文，不得出现英文句子
2. 禁止使用任何数学公式或 LaTeX 符号，用中文文字描述代替
3. 禁止使用 Markdown 代码块或特殊符号
4. 直接输出纯文本段落，结构：①新颖性结论 ②最相关论文列表（含链接）③相似点与不同点分析
""")]


def reviewer_fast_prompt(state):
    image_parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
        for img in state['referee']['images']
    ]

    prompt = [{"type": "text", "text": r"""你是一位严格的科学审稿人。请阅读下方 LaTeX 格式的科学论文，撰写一份详细的审稿报告，包括论文的亮点与不足，并对不足之处提出具体的改进建议。

审稿时请重点关注：
- 找出论文中所有缺陷与问题
- 指出可能不正确或不严谨的地方
- 指出需要进一步修改才能提升质量的地方
- 仔细核查论文中的证据是否充分支持结论
- 若结果不佳，判断是研究策略失误还是意料之外的发现

请判断该论文是否值得发表，并给出 0（极差）到 9（极好）的评分。

**请严格按以下格式回复**：
\begin{{REVIEW}}
<在此填写审稿报告>
\end{{REVIEW}}
"""}] + image_parts

    return [HumanMessage(content=prompt)]
