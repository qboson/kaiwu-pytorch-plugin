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
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "torchvision",
            "pylint>=2.17.5",
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
            "matplotlib>=3.5.0",
            "scikit-learn>=1.0.0",
        ],
    },
    keywords="quantum computing, boltzmann machine, restricted boltzmann machine, pytorch, "
    "machine learning, deep learning",
    license="Apache License 2.0",
    zip_safe=False,
)
