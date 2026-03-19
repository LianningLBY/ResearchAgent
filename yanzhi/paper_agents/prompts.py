from langchain_core.messages import HumanMessage, SystemMessage


def abstract_prompt(state, attempt):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""
第 {attempt} 次尝试。

根据以下研究想法、方法和结果，为一篇科学论文撰写标题和摘要。请遵循以下要求：
- 简要描述研究问题
- 简要说明解决思路
- 提及所用数据集和方法
- 简要描述研究结果
- 用 LaTeX 书写摘要
- 摘要中不写公式或引用
- 摘要为单段落，不分节、不换行

研究想法：
{state['idea']['Idea']}

研究方法：
{state['idea']['Methods']}

研究结果：
{state['idea']['Results']}

**请严格按以下格式回复**

```json
{{"Title": "论文标题", "Abstract": "论文摘要"}}
```
""")]


def abstract_reflection(state):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""根据研究想法、方法和结果，以及之前撰写的摘要，重新改写摘要使其更清晰。

研究想法：
{state['idea']['Idea']}

研究方法：
{state['idea']['Methods']}

研究结果：
{state['idea']['Results']}

之前的摘要：
{state['paper']['Abstract']}

**请严格按以下格式回复**

\begin{{Abstract}}
<摘要内容>
\end{{Abstract}}

要求：单段落，不分节，简洁清晰，逻辑连贯。
""")]


def introduction_prompt(state):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""根据以下论文标题、摘要、研究想法和方法，用 LaTeX 撰写论文引言。

论文标题：
{state['paper']['Title']}

论文摘要：
{state['paper']['Abstract']}

研究想法：
{state['idea']['Idea']}

研究方法：
{state['idea']['Methods']}

请按以下格式回复：

\begin{{Introduction}}
<引言内容>
\end{{Introduction}}

引言写作要求：
- 用 LaTeX 书写
- 说明研究问题及其难点
- 说明本文如何尝试解决该问题
- 说明如何验证解决效果
- 不分小节
- 不添加参考文献
- 不定义自定义命令
- 行文流畅、有充分动机铺垫
""")]


def introduction_reflection(state):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""根据论文标题、摘要、研究想法和方法，改写引言使其更清晰。

论文标题：
{state['paper']['Title']}

论文摘要：
{state['paper']['Abstract']}

研究想法：
{state['idea']['Idea']}

研究方法：
{state['idea']['Methods']}

之前的引言：
{state['paper']['Introduction']}

请按以下格式回复：

\begin{{Introduction}}
<引言内容>
\end{{Introduction}}

要求同上。
""")]


def methods_prompt(state):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""根据以下论文标题、摘要、引言和方法简述，撰写详细的方法章节。

论文标题：
{state['paper']['Title']}

论文摘要：
{state['paper']['Abstract']}

论文引言：
{state['paper']['Introduction']}

方法简述：
{state['idea']['Methods']}

请按以下格式回复：

\begin{{Methods}}
<方法内容>
\end{{Methods}}

要求：
- 用 LaTeX 书写
- 详细描述所用方法、数据集、评估指标等
- 不写参考文献（后续统一添加）
- 不定义自定义命令
- 与引言内容自然衔接
- 小节标题首字母大写，不全部大写
- 可写小节和子小节，但不写顶级节（section）
""")]


def results_prompt(state):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""根据论文标题、摘要、引言、方法和结果简述，撰写详细的结果与讨论章节。

论文标题：
{state['paper']['Title']}

论文摘要：
{state['paper']['Abstract']}

论文引言：
{state['paper']['Introduction']}

论文方法：
{state['paper']['Methods']}

结果简述：
{state['idea']['Results']}

请按以下格式回复：

\begin{{Results}}
<结果内容>
\end{{Results}}

要求：
- 用 LaTeX 书写
- 详细阐述和解读研究结果
- 不添加图片占位符（图片后续单独插入）
- 描述从结果中学到的内容
- 不写参考文献
- 不定义自定义命令
- 小节标题首字母大写
- 可写小节和子小节，但不写顶级节
- 末尾可做小结，但不写结论小节
""")]


def refine_results_prompt(state):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=fr"""你将收到一个包含文字和图片的结果章节。文字和图片是独立添加的，可能缺乏有机整合。

请重写文字使其与图片及图注更加连贯。规则如下：

- **不得删除任何图片，所有图片必须保留**
- 使用 Figure \ref{{fig:...}} 语法添加适当的图片引用
- 调整文字顺序和内容，提升清晰度和流畅性
- 仅在有助于提升清晰度时才重新排序图片和段落
- 不删除技术或科学内容
- 用 LaTeX 书写
- 小节标题首字母大写

结果章节：
{state['paper']['Results']}

**请严格按以下格式回复**：

\begin{{Results}}
<结果内容>
\end{{Results}}
""")]


def conclusions_prompt(state):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""根据以下论文内容，撰写结论章节。

论文标题：
{state['paper']['Title']}

论文摘要：
{state['paper']['Abstract']}

论文引言：
{state['paper']['Introduction']}

论文方法：
{state['paper']['Methods']}

研究结果：
{state['paper']['Results']}

要求：
- 用 LaTeX 书写
- 简要描述研究问题和本文的解决思路
- 描述所用数据集和方法
- 描述研究结果
- 总结从结果和本研究中获得的认识
- 不添加参考文献
- 不定义自定义命令
- 小节标题首字母大写

请按以下格式回复：

\begin{{Conclusions}}
<结论内容>
\end{{Conclusions}}
""")]


def caption_prompt(state, image, name=None):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=[
            {"type": "text", "text": rf"""请为科学论文中的一幅图片撰写图注。

要求：
- 用 LaTeX 书写
- 描述图片所展示的内容
- 结合下方的结果上下文，将图注与研究内容关联
- 不引用任何章节或小节
- 尽量描述从图中能学到什么
- 图注尽量简洁，但内容完整

结果上下文：
{state['idea']['Results']}

**请严格按以下格式回复**

\begin{{Caption}}
<图注内容>
\end{{Caption}}
"""},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
        ])
    ]


def plot_prompt(state, images):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""请将一组图片插入论文的结果章节。你已有当前结果章节文本，以及包含图片名称和图注的字典。请将这些图片插入文本中最合适的位置，并附上图注。如果文本中已有部分图片，请勿删除或移动原有图片，只添加新图片。

结果章节：
{state['paper']['Results']}

图片字典：
{images}

请按以下格式回复：

\begin{{Section}}
<章节内容>
\end{{Section}}

图片路径格式："../input_files/plots/图片名"，图片宽度为半页宽。为每张图片根据图注选择合适的标签。所有内容须兼容 LaTeX。图注须为单段落，不使用 enumerate 或 itemize。
""")]


def LaTeX_prompt(text):
    return [HumanMessage(content=fr"""对以下文本进行最小限度的修改，使其兼容 LaTeX。例如：

- Subhalo\_A → Subhalo\ensuremath{{\_}}A
- Eisenstein & Hu → Eisenstein \& Hu

特别注意下划线 \_：
- 若在公式中，不修改
- 若在图片路径中，不修改
- 若在引用标签中，不修改
- 其他情况改为 \ensuremath{{\_}}
- 注意 % 符号：在 LaTeX 中若未转义将注释该行后所有内容
- 确保行内公式用 $ 包裹

原始文本：
{text}

**请按以下格式回复**：

\begin{{Text}}
<文本内容>
\end{{Text}}
""")]


def clean_section_prompt(state, text):
    return [HumanMessage(content=fr"""你将收到一段 LaTeX 文本。请做最小限度的清理，确保结果为合法 LaTeX 且保留原始含义。

允许做的：
- 拆分过长段落
- 调整宽表格为全页宽
- 将表格或图片内的引用移到外部（LaTeX 不允许在浮动体内使用 \citep 等命令）

**不允许做的**：
- 改变段落或图片的顺序
- 新增章节或重构内容
- 删除或改写超出上述范围的内容

确保修改后的输出可以在 LaTeX 中正常编译。

---

**原始文本：**
{text}

---

**请严格按以下格式回复**：

\begin{{Text}}
<清理后的 LaTeX 文本>
\end{{Text}}
""")]


def summary_prompt(state, text, summary):
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""对以下文本进行摘要，并与已有的摘要合并。

已有摘要：
{summary}

待摘要文本：
{text}

请按以下格式回复：
\begin{{Summary}}
<合并后的摘要>
\end{{Summary}}
""")]


def references_prompt(state, text):
    return [HumanMessage(content=rf"""你将收到一段科学论文 LaTeX 文本，其中包含图片和图片引用。请确保图片引用正确，如有错误请修正。规则：
- 不增删文本
- 专注于修正图片引用错误
- 若引用与对应图片标签匹配，无需修改

原始文本：
{text}

**请按以下格式回复**

\begin{{Text}}
<修正后的文本>
\end{{Text}}
""")]


def fixer_prompt(text, section_name):
    return [HumanMessage(content=fr"""从以下文本中提取 {section_name} 章节内的所有内容。

文本：
{text}

请按以下格式回复：

\begin{{{section_name}}}
<{section_name}>
\end{{{section_name}}}

提取内容中不得包含以下行：

```latex
\documentclass{{article}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\begin{{document}}
\section{{Results}}
\end{{document}}
```
""")]


def fix_latex_bug_prompt(state):
    with open(state['files']['LaTeX_err'], 'r') as f:
        error = f.read()
    return [HumanMessage(content=fr"""以下文本存在 LaTeX 编译错误，请修复使其能正常编译。注意：

- 此文本是论文的一个章节，无需添加 \begin{{document}} 等
- 修复**所有** LaTeX 错误
- 特别注意下划线 _ 可能需要改为 \_
- 保持文本内容不变，只修复错误

文本：
{state['paper'][state['latex']['section_to_fix']]}

错误信息：
{error}

请按以下格式回复：

\begin{{Text}}
<修复后的文本>
\end{{Text}}
""")]


def keyword_prompt(state, keywords_list=None):
    """生成关键词（中文版使用 LLM 直接提取，不依赖 AAS 关键词库）"""
    return [
        SystemMessage(content=f"你是一位{state['writer']}。"),
        HumanMessage(content=rf"""根据以下研究想法和方法，提取 5-8 个中英文关键词，用于学术论文。

研究想法：
{state['idea']['Idea']}

研究方法：
{state['idea']['Methods']}

要求：
- 关键词应准确反映研究主题
- 可以是中文或英文（学术领域通用术语可保留英文）
- 用逗号分隔

**请按以下格式回复**

\begin{{Keywords}}
<关键词列表，用逗号分隔>
\end{{Keywords}}
""")]
