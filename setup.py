from setuptools import setup

deps = []
with open("./requirements.txt") as f:
    for line in f.readlines():
        if not line.strip():
            continue
        deps.append(line.strip())

setup(
    name="graphrag-lite",
    version="0.1.0",
    author="Hongyang Yu",
    author_email="hongyang.yu@coupa.com",
    description="A light-weight and naive Python implementation of the GraphRAG algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["graphrag_lite"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version required
    install_requires=deps,
)
