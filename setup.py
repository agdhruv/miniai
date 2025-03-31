from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="miniai",
    version="0.1.0",
    author="Dhruv Agarwal",
    author_email="dhruv_agarwal@outlook.com",
    description="A minimalist Python library for byte-sized AI tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agdhruv/MiniAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",  # Optional
        "anthropic>=0.3.0",  # Optional
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
