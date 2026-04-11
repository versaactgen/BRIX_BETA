from setuptools import setup, find_packages

setup(
    name="brix",
    version="0.1.0",
    description="A modular Reinforcement Learning framework.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=2.0.0",
        "gymnasium>=0.28.1",
    ],
    python_requires=">=3.8",
)
