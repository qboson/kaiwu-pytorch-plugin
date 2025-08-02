FROM python:3.10

# Use Aliyun apt mirror
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# Install LaTeX
RUN apt-get update && \
    apt-get install -y texlive-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
COPY requirements ./requirements

RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple -r requirements/devel.txt