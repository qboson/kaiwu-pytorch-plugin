from setuptools import setup, find_namespace_packages


# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# 读取requirements文件
def read_requirements():
    with open("requirements/requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="kaiwu-torch-plugin",
    version="0.1.0",
    author="QBoson Inc",
    author_email="developer@boseq.com",
    description="A PyTorch plugin for training and evaluating Restricted Boltzmann Machines (RBM) and "
    "Boltzmann Machines (BM) with quantum computing support",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/QBoson/kaiwu-pytorch-plugin",
    project_urls={
        "Bug Reports": "https://github.com/QBoson/kaiwu-pytorch-plugin/issues",
        "Source": "https://github.com/QBoson/kaiwu-pytorch-plugin",
        "Documentation": "https://github.com/QBoson/kaiwu-pytorch-plugin#readme",
    },
    packages=find_namespace_packages(include=["kaiwu.*"]),
    install_requires=read_requirements(),
    keywords="quantum computing, boltzmann machine, restricted boltzmann machine, pytorch, "
    "machine learning, deep learning",
    license="Apache License 2.0",
    zip_safe=False,
)
