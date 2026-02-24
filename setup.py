from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


VERSION = read_text(ROOT / "VERSION").strip()
README = read_text(ROOT / "README.md")


setup(
    name="nanohubsubmit",
    version=VERSION,
    description="Modern standalone NanoHUB submit client.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="NanoHUB Submit Team",
    python_requires=">=3.7",
    packages=find_packages(include=["nanohubsubmit", "nanohubsubmit.*"]),
    include_package_data=True,
    install_requires=[
        "importlib-metadata>=4.0; python_version < '3.8'",
    ],
    entry_points={
        "console_scripts": [
            "nanohub-submit=nanohubsubmit.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["nanohub", "submit", "client", "workflow"],
    project_urls={
        "Repository": "https://github.com/nanohub/submit",
    },
)
