"""实验模块专用提示词。"""


def data_requirements_prompt(methods: str) -> str:
    """从研究方法中提取数据需求，生成数据集搜索词。"""
    return f"""你是一位数据科学家。请阅读以下研究方法，提取对数据集的核心需求，并生成合适的搜索词。

## 研究方法
{methods}

## 输出格式（严格 JSON）
```json
{{
  "task_type": "classification 或 regression 或 clustering 或 nlp 或 其他",
  "domain": "研究领域，中文，10字以内（如：网络安全、医疗影像、金融时序）",
  "search_query": "用于搜索数据集的英文关键词，3-5词，空格分隔",
  "needs_local_data": true 或 false,
  "reason": "为什么需要这类数据，中文，30字以内"
}}
```

## 关于 needs_local_data 的判断规则
- **绝大多数情况下应设为 false**，因为 HuggingFace、Papers With Code 上有大量公开数据集
- 网络安全（KDD99、CICIDS、UNSW-NB15）、医疗（MIMIC、PhysioNet）、NLP（各类语料库）均有公开数据集，应设 false
- **只有在以下情况才设 true**：研究明确依赖某家公司/机构的专有内部数据，且该数据完全无公开替代品（例如"使用某银行内部交易流水"）
- 如果不确定，请设 false，系统会自动搜索合适的公开数据集
"""


def criteria_infer_prompt(methods: str, data_description: str) -> str:
    return f"""你是一位科研方法论专家。请根据以下研究方法，推断本次实验的合理验收标准。

## 数据和工具描述
{data_description}

## 研究方法
{methods}

## 要求
1. 标准必须基于研究目标，不能脱离实际（如社会科学数据 R²=0.3 可能已经很好）。
2. 给出 2-4 条具体、可观测的标准（优先用可量化指标）。
3. 最后一条标准建议是：「代码能完整运行并产生有意义的输出」（作为最低要求）。
4. 用中文输出，直接列出标准，不加额外解释。

## 输出示例
- 主要自变量的回归系数 p < 0.05
- 模型 R² ≥ 0.3
- 至少输出 1 张展示变量关系的图表
- 代码能完整运行并产生有意义的输出
"""


def code_gen_prompt(data_description: str, methods: str,
                    criteria: str = "",
                    prev_output: str = "", prev_code: str = "") -> str:
    history_section = ""
    if prev_output:
        history_section = f"""
## 上次执行结果（需改进）
```
{prev_output[:3000]}
```

## 上次代码
```python
{prev_code[:2000]}
```

请根据上述结果**修复问题或改进**代码。
"""

    criteria_section = f"\n## 验收标准（代码输出需满足）\n{criteria}\n" if criteria else ""

    return f"""你是一位数据科学家。请根据以下研究信息，生成可直接运行的 Python 实验代码。

## 数据和工具描述
{data_description}

## 研究方法
{methods}
{criteria_section}{history_section}
## 要求
1. 代码完整、可直接运行，不需要用户手动修改变量。
2. 「数据和工具描述」中若有「本地路径」，直接使用该路径读取数据；若注明「合成数据」，则用 numpy/pandas 生成符合研究需求的模拟数据。
3. 使用 `plt.show()` 展示图表（系统已自动将其重定向为保存文件）。
4. 只使用常见包（numpy, pandas, matplotlib, scipy, sklearn, statsmodels 等）。
5. 在代码末尾打印关键指标（如均值、标准差、R²、p 值等）。
6. 不要使用 `input()`，不要向项目目录外写文件。

## 输出格式（严格包裹，不输出其他内容）
\\begin{{CODE}}
# 你的 Python 代码
\\end{{CODE}}
"""


def fix_code_prompt(code: str, exec_output: str) -> str:
    return f"""你是一位 Python 调试专家。请修复以下代码中的错误。

## 错误的代码
```python
{code[:3000]}
```

## 执行错误输出
```
{exec_output[:2000]}
```

## 要求
1. 只修复错误，不改变算法逻辑。
2. 如果是缺少数据文件，改为生成合成数据演示。
3. 如果是缺少依赖包，改用等价的常见包实现。

## 输出格式（严格包裹）
\\begin{{CODE}}
# 修复后的完整代码
\\end{{CODE}}
"""


def diagnose_prompt(exec_output: str, methods: str, criteria: str,
                    exec_success: bool) -> str:
    run_status = "代码运行成功但结果不满足验收标准" if exec_success else "代码运行失败"
    return f"""你是一位数据科学家。请诊断以下实验结果的问题，并生成文献检索词。

## 运行状态
{run_status}

## 验收标准
{criteria}

## 研究方法
{methods}

## 实验输出
```
{exec_output[:3000]}
```

## 输出格式（严格 JSON）
```json
{{
  "failure_type": "code_error 或 insufficient",
  "diagnosis": "具体问题描述，中文，50字以内",
  "search_query": "针对该问题的英文文献检索词（3-5个关键词，用空格分隔）"
}}
```

注意：failure_type 为 code_error 说明代码本身有 bug；insufficient 说明代码能运行但结果质量不足。
"""


def method_refine_prompt(methods: str, diagnosis: str,
                          literature_found: str, human_input: str = "") -> str:
    human_section = f"\n## 用户额外指示\n{human_input}\n" if human_input else ""
    return f"""你是一位科研方法论专家。请基于文献，针对诊断出的问题改进研究方法。

## 当前研究方法
{methods}

## 诊断的问题
{diagnosis}

## 找到的相关文献
{literature_found[:3000]}
{human_section}
## 要求
1. 改进要有文献依据，明确指出参考了哪篇文献的哪个方法。
2. 只针对诊断出的问题改进，保留其他部分不变。
3. 输出完整的优化后方法，用以下标记包裹：

\\begin{{METHODS}}
（优化后的完整研究方法）
\\end{{METHODS}}
"""


def results_summary_prompt(exec_output: str, methods: str,
                            data_description: str) -> str:
    return f"""你是一位科研写作助手。请根据实验输出，生成规范的实验结果摘要，用于论文的"结果"章节。

## 研究方法
{methods}

## 实验输出
```
{exec_output[:4000]}
```

## 要求
1. 用中文撰写，300-500 字。
2. 报告关键数值指标（均值、标准差、显著性等）。
3. 描述图表内容（如有）。
4. 避免引入实验输出中没有的数据。
5. 直接输出摘要文本，不需要额外标记。
"""
