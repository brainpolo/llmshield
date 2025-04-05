"""
Setup configuration for llmshield package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llmshield",
    packages=find_packages(exclude=["tests", "tests.*"]),
    # Development Dependencies
    extras_require={
        "dev": [
            "black>=22.0.0",
            "isort>=5.0.0",
            "pylint>=3.3.6",
            "coverage>=7.8.0",
            "build>=1.2.2",
            "packaging>=24.2",
            "pyproject_hooks>=1.2.0",
            "setuptools>=75.8.0",
        ],
    },
    # Metadata
    author="Aditya Dedhia",
    author_email="adityadedhia@hey.com",
    description="Shields your confidential data from third party LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brainpolo/llmshield",
    project_urls={
        "Bug Tracker": "https://github.com/brainpolo/llmshield/issues",
        "Documentation": "https://llmshield.readthedocs.io/",
    },
    # Classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Requirements
    python_requires='>=3.10',
)