"""
Setup script for Claude Enterprise Trading
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line.split(">=")[0].split("==")[0])

setup(
    name="claude-enterprise-trading",
    version="1.0.0",
    author="OpenClaw Community",
    author_email="community@openclaw.ai",
    description="Agentic trading infrastructure using Claude Enterprise + OpenClaw",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openclaw/claude-enterprise-trading",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
        ],
        "enterprise": [
            "factset-sdk",
            "msci-sdk",
        ]
    },
    entry_points={
        "console_scripts": [
            "claude-trading=src.orchestrator:main",
            "strategy-builder=src.strategy.nl_to_tree:main",
            "backtest-runner=src.backtest.engine:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)