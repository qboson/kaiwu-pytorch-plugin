# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from kaiwu.torch_plugin import __version__

project = 'Kaiwu Pytorch Plugin'
copyright = '2025 Beijing QBoson Quantum Technology Co., Ltd'
author = 'QBoson Inc'
release = __version__
version = __version__

# language 默认语言
language = 'zh_CN'

# 启用 gettext
locale_dirs = ['locale/']
gettext_compact = False

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinxcontrib.jquery',
    'sphinx.ext.imgmath',
    'myst_parser',
    'sphinxcontrib.mermaid'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md', 'source/getting_started/start.rst']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = "_static/sdk-logo.png"
html_favicon = "_static/sdk-logo.png"

html_theme_options = {
    "show_nav_level": 1,
    "logo": {
        "text": project,
        "image_dark": "",
        "image_light": ""
    },
    # 导航栏配置
    "navbar_center": ["navbar-nav"],  # 中间导航链接
    "navbar_persistent": ["search-button"],  # 常驻元素（如搜索按钮）
    # 页脚配置
    "footer_start": ["copyright"],  # 页脚开头
    "footer_end": ["theme-version"],  # 页脚结尾
    "show_toc_level": 2,  # 侧边栏目录显示层级
}

html_show_sourcelink = False
html_css_files = ['custom.css']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
