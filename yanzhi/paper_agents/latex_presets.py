from .journal import Journal, LatexPresets

# 标准中文学术格式（ctexart 单栏）
latex_none = LatexPresets(
    article="ctexart",
    usepackage=r"\usepackage[margin=2.5cm]{geometry}",
    abstract=lambda x: f"\\maketitle\n\\begin{{abstract}}\n{x}\n\\end{{abstract}}\n",
    bibliographystyle=r"\bibliographystyle{unsrt}",
)

# 中国知网风格（双栏）
latex_cnki = LatexPresets(
    article="ctexart",
    layout="twocolumn",
    usepackage=(
        r"\usepackage[margin=2cm]{geometry}" "\n"
        r"\usepackage{multicol}"
    ),
    abstract=lambda x: f"\\maketitle\n\\begin{{abstract}}\n{x}\n\\end{{abstract}}\n",
    keywords=lambda x: f"\\noindent\\textbf{{关键词：}}{x}\n",
    bibliographystyle=r"\bibliographystyle{unsrt}",
)

# IEEE 双栏（含 ctex）
latex_ieee = LatexPresets(
    article="IEEEtran",
    usepackage=(
        r"\usepackage{ctex}" "\n"
        r"\usepackage{cite}" "\n"
        r"\usepackage{url}"
    ),
    abstract=lambda x: (
        f"\\maketitle\n"
        f"\\begin{{abstract}}\n{x}\n\\end{{abstract}}\n"
    ),
    keywords=lambda x: f"\\begin{{IEEEkeywords}}\n{x}\n\\end{{IEEEkeywords}}\n",
    bibliographystyle=r"\bibliographystyle{IEEEtran}",
)

journal_dict = {
    Journal.NONE: latex_none,
    Journal.CNKI: latex_cnki,
    Journal.IEEE: latex_ieee,
}
