from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="inference_kit",
    version="0.1.0",
    author="Chris Nippert",
    author_email="contact@chrisnippert.com",
    description="Utility functions for working with causal language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chrisnippert/inference_kit",
    packages=find_packages(),
    python_requires=">=3.13",
    install_requires=requirements,
    extras_require={
        "cpu": ["torch>=2.0.0"],
        "cuda": ["torch>=2.0.0"],  # Users should install from pytorch.org
    },
)
