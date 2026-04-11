from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
REQUIREMENTS = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="brix",
    version="0.1.0",
    description="A modular Reinforcement Learning framework.",
    author="Your Name",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.10",
)
