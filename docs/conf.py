# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

from sphinx import addnodes

sys.path.insert(0, os.path.abspath("../src"))
from kaiwu.torch_plugin import __version__

project = "Kaiwu-PyTorch-Plugin"
copyright = "2025 Beijing QBoson Quantum Technology Co., Ltd"
author = "QBoson Inc"
release = __version__
version = __version__

# Read the Docs 为每个语言项目提供该环境变量；本地构建时默认使用中文。
# 中文翻译目录使用 Sphinx 与 Read the Docs 共同使用的 zh_CN 语言代码。
rtd_language = os.environ.get("READTHEDOCS_LANGUAGE", "zh_CN")
language = "zh_CN" if rtd_language in ("zh", "zh-cn") else rtd_language

# 启用 gettext
locale_dirs = ["locale/"]
gettext_compact = False

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinxcontrib.jquery",
    # "sphinx.ext.imgmath",  # 不要同时启用
    # "sphinx.ext.mathjax",  # 启用 MathJax
    "sphinxcontrib.katex",   # 或启用 KaTeX，但不要同时启用两者
    "myst_parser",
    "sphinx.ext.napoleon",
    'sphinxcontrib.mermaid',  # 启用 Mermaid 扩展
    "sphinxcontrib.bibtex",   # 启用 BibTeX 支持
]
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence", 
    "deflist", 
    "substitution", 
    "tasklist", 
]
myst_fence_as_directive = ["mermaid"]
katex_prerender = True

# 公式编号：启用 Sphinx numfig，使公式编号全局唯一（跨页面连续递增），
# 避免每个页面从 (1) 重新计数导致跨页引用歧义。
numfig = True
math_numfig = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "source/getting_started/start.md",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/sdk-logo.png"
html_favicon = "_static/sdk-logo.png"

html_theme_options = {
    "show_nav_level": 1,
    "logo": {"text": project, "image_dark": "", "image_light": ""},
    # 导航栏配置
    "navbar_center": ["navbar-nav"],  # 中间导航链接
    "navbar_persistent": ["search-button"],  # 常驻元素（如搜索按钮）
    # 页脚配置
    "footer_start": ["copyright"],  # 页脚开头
    "footer_end": ["theme-version"],  # 页脚结尾
    "show_toc_level": 2,  # 侧边栏目录显示层级
}

html_show_sourcelink = False
html_css_files = ["custom.css"]


# -- Options for BibTeX -----------------------------------------------------------
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'   # 参考文献样式，可选：'plain', 'unsrt', 'alpha', 'unsrtalpha'
bibtex_reference_style = 'label' # 引用样式，可选：'label', 'author_year', 'super'
# 可选：如果需要中文编码，请保持默认，或根据 .bib 文件的实际编码设置
# bibtex_encoding = 'utf-8-sig'


def _hide_attributes_from_page_toc(app, doctree):
    """保留属性正文，但不将属性加入页面右侧目录。"""
    del app
    for description in doctree.findall(addnodes.desc):
        if description.get("objtype") == "attribute":
            description["no-contents-entry"] = True


def setup(app):
    """注册页面目录过滤器。"""
    app.connect("doctree-read", _hide_attributes_from_page_toc, priority=400)

# arXiv and GitHub intermittently terminate TLS connections from the local
# linkcheck client. The corresponding public pages are verified separately.
linkcheck_ignore = [
    r"https://arxiv\.org/abs/2508\.11190",
    r"https://github\.com/QBoson/Kaiwu-pytorch-plugin/issues",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
