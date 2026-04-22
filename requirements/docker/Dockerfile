FROM quay.io/jupyter/minimal-notebook:f3079808ca8c

# 切换到 root 用户以便安装系统包
USER root

# 安装编译工具及可能需要的其他依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # 以下按需添加，常见于科学计算/数据处理包的依赖
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 切换回普通用户（jovyan）执行后续操作
USER jovyan

# 复制 requirements.txt 并安装
COPY ../requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制 whl 文件并安装
COPY ../kaiwu-1.3.1-py3-none-any.whl kaiwu-1.3.1-py3-none-any.whl
RUN pip install --no-cache-dir kaiwu-1.3.1-py3-none-any.whl \
    && rm kaiwu-1.3.1-py3-none-any.whl

# 如果需要，可以在此处清理编译工具以减小镜像体积（可选）
# USER root
# RUN apt-get purge -y build-essential && apt-get autoremove -y
# USER jovyan