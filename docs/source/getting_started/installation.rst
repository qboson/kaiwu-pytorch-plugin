========
安装指南
========

本章介绍如何安装 Kaiwu-PyTorch-Plugin 及其依赖项。

1. 环境要求
-----------

在安装之前，请确保您的系统满足以下要求：

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 依赖项
     - 版本要求
   * - Python
     - 3.10
   * - PyTorch
     - 2.7.0
   * - NumPy
     - 2.2.6
   * - Kaiwu SDK
     - v1.2.0+

检查 Python 版本：

.. code-block:: bash

    python --version
    # or
    python3 --version

如果需要安装 Python 3.10，请访问 `Python 3.10 下载页面 <https://www.python.org/downloads/release/python-31011/>`_。

2. 安装 Kaiwu-PyTorch-Plugin
-----------------------------

2.1 创建并激活环境
^^^^^^^^^^^^^^^^^^

推荐使用 conda 创建独立的 Python 环境：

.. code-block:: bash

    # Create a new environment
    conda create -n quantum_env python=3.10

    # Activate the environment
    conda activate quantum_env

2.2 克隆仓库
^^^^^^^^^^^^

从 GitHub 克隆项目到本地：

.. code-block:: bash

    git clone https://github.com/QBoson/Kaiwu-pytorch-plugin.git
    cd kaiwu-pytorch-plugin

2.3 安装依赖
^^^^^^^^^^^^

安装项目依赖：

.. code-block:: bash

    pip install -r requirements/requirements.txt

2.4 安装插件
^^^^^^^^^^^^

.. code-block:: bash

    pip install .

3. 安装 Kaiwu SDK（必需）
-------------------------

Kaiwu-PyTorch-Plugin 依赖 Kaiwu SDK 提供量子计算能力，您需要单独安装 Kaiwu SDK。

3.1 获取 SDK
^^^^^^^^^^^^

1. 访问 `Kaiwu SDK 下载页面 <https://platform.qboson.com/sdkDownload>`_
2. 下载适合您系统的 SDK 安装包
3. 参考 `Kaiwu SDK 安装说明 <https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/sdk_installation_instructions.html>`_ 以完成安装

3.2 配置授权信息
^^^^^^^^^^^^^^^^

安装完成后，您需要配置 SDK 授权信息：

::

    User ID: <your-user-id>
    SDK Token: <your-sdk-token>

.. note::

    请将上述信息替换为您的实际授权信息。授权信息可在 QBoson 平台的 `Kaiwu SDK 页面 <https://platform.qboson.com/>`_ 获取。

4. 验证安装
----------------------

安装完成后，运行以下代码验证安装是否成功：

.. code-block:: python

    # 验证 PyTorch
    import torch
    print(f"PyTorch version: {torch.__version__}")



    # 验证 Kaiwu SDK
    import kaiwu
    import numpy as np
    from kaiwu.classical import SimulatedAnnealingOptimizer
    opt = SimulatedAnnealingOptimizer()
    mat = np.array([[1, -1], [-1, 1]])
    result = opt.solve(mat)
    print(f"Kaiwu SDK version: {kaiwu.__version__}")
    print(result)

    # 验证 Kaiwu-PyTorch-Plugin
    from kaiwu.torch_plugin import RestrictedBoltzmannMachine
    print("Kaiwu-PyTorch-Plugin imported successfully!")
    # 简单测试
    rbm = RestrictedBoltzmannMachine(num_visible=10, num_hidden=5)
    print(f"RBM created with {rbm.num_visible} visible and {rbm.num_hidden} hidden units")

如果没有报错，至此您已经安装成功。后续您可以根据需求进行模型的构建，并使用经典计算器进行模型验证，验证通过后再根据如下步骤切换到量子计算器以利用量子计算资源。

5. 获取量子计算机访问
---------------------

要体验真正的量子计算能力，您需要获取量子计算机的访问权限：

1. 在 `QBoson 平台 <https://platform.qboson.com/>`_ 注册账号
2. 通过平台联系官方工作人员获取真机配额

.. note::

    在获取真机访问权限之前，您可以使用模拟器进行开发和测试验证。Kaiwu SDK 提供了多种经典优化器（如模拟退火优化器）作为量子采样器的经典替代方案。

6. 开发环境设置（可选）
-----------------------

如果您计划参与插件的开发，可以安装开发依赖：

.. code-block:: bash

    pip install -r requirements/devel.txt

运行测试：

.. code-block:: bash

    # Run all tests
    pytest tests/

    # Run specific tests
    pytest tests/test_rbm.py

代码风格检查：

.. code-block:: bash

    pylint kaiwu/

7. 常见问题
-----------

Q: 安装时提示 Python 版本不兼容？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: Kaiwu-PyTorch-Plugin 目前基于 Python 3.10。请使用 conda 创建 Python 3.10 环境：

.. code-block:: bash

    conda create -n quantum_env python=3.10
    conda activate quantum_env

Q: 无法导入 kaiwu.torch_plugin？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: 请确保：

1. 当前环境已激活（``conda activate quantum_env``）
2. Kaiwu SDK 已正确安装
3. 已正确安装 Kaiwu-PyTorch-Plugin（``pip install .``），检查是否已经安装可以使用
    ``pip list`` 或者 ``pip show kaiwu-torch-plugin``


Q: 如何更新到最新版本？
^^^^^^^^^^^^^^^^^^^^^^^

A: 进入项目目录并拉取最新代码：

.. code-block:: bash

    cd kaiwu-pytorch-plugin
    git pull origin main
    pip install .

