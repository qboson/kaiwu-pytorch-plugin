# Kaiwu Pytorch Plugin 文档框架

本文档使用 Sphinx 文档框架搭建，参考了 kaiwu_community 项目的文档结构。

## 文档框架特性

- **文档框架**: Sphinx 8.1.3
- **主题**: PyData Sphinx Theme 0.15.4
- **语言**: 中文 (zh_CN)
- **支持格式**: Markdown、TXT
- **扩展功能**:
  - 自动生成 API 文档 (sphinx.ext.autodoc)
  - 查看源代码 (sphinx.ext.viewcode)
  - 数学公式支持 (sphinx.ext.imgmath)
  - Markdown 支持 (myst-parser)
  - 中文搜索支持 (jieba)

## 文档目录结构

```
docs/
├── conf.py                      # Sphinx 配置文件
├── index.md                    # 文档首页
├── Makefile                     # 构建脚本
├── make.bat                     # Windows 构建脚本
├── _static/                     # 静态资源
│   ├── custom.css              # 自定义样式
│   └── sdk-logo.png            # 项目 Logo
└── source/                      # 文档源文件
    ├── getting_started/         # 入门指南
    │   ├── index.md
    │   ├── introduction.md
    │   ├── installation.md
    │   └── quickstart.md
    ├── advanced/                # 进阶知识
    │   ├── index.md
    │   └── advanced_features.md
    ├── modules/                 # 模块手册
    │   ├── index.md
    │   └── kaiwu.torch_plugin.md
    ├── faq/                     # 常见问题
    │   ├── index.md
    │   └── faq.md
    └── about/                   # 关于
        ├── index.md
        ├── release_notes.md
        └── license.md
```

## 如何构建文档

### 1. 安装依赖

```bash
pip install -r requirements/devel.txt
```

### 2. 构建 HTML 文档

```bash
cd docs
make html
```

构建完成后，文档会生成在 `docs/_build/html/` 目录中。

### 3. 查看文档

在浏览器中打开 `docs/_build/html/index.html` 文件即可查看文档。

### 4. 清理构建文件

```bash
cd docs
make clean
```

## Read the Docs 部署

项目已配置 `.readthedocs.yaml` 文件，可以直接在 Read the Docs 平台上部署。

配置要点：
- Python 版本: 3.10
- 构建系统: Ubuntu 22.04
- 依赖文件: requirements/devel.txt
- Sphinx 配置: docs/conf.py

## 文档编写指南

### 添加新文档

1. 在相应的目录下创建 `.md` 文件
2. 在对应的 `index.md` 文件中添加引用
3. 重新构建文档

### 文档格式

支持以下格式：
- **Markdown**: 通过 myst-parser 支持
- **TXT**: 作为 Markdown 处理

### API 文档

API 文档通过 `sphinx.ext.autodoc` 自动生成，只需在 Markdown 文件中使用：

````markdown
```{automodule} kaiwu.torch_plugin
:members:
:undoc-members:
:show-inheritance:
```
`````

## 待完成事项

以下内容需要后续补充：

1. **入门指南**
   - [ ] 完善项目简介 (introduction.md)
   - [ ] 完善安装指南 (installation.md)
   - [ ] 完善快速开始 (quickstart.md)

2. **进阶知识**
   - [ ] 添加高级功能文档 (advanced_features.md)

3. **常见问题**
   - [ ] 补充常见问题解答 (faq.md)

4. **关于**
   - [ ] 更新发行说明 (release_notes.md)

## 注意事项

1. 文档构建时可能会出现一些警告，这些主要是内容相关的问题，不影响框架本身
2. 图片路径需要相对于文档根目录
3. 建议使用 MyST Markdown 格式编写文档，以获得更好的 Sphinx 支持
