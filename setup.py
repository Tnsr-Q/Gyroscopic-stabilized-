#!/usr/bin/env python3
"""
Setup script for the Gyroscopic Stabilized Quantum Engine.

This provides a minimal setup.py for the modular quantum engine,
allowing for proper package installation and distribution.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gyroscopic-stabilized-engine",
    version="1.0.0",
    author="Tnsr-Q",
    author_email="quantquiplabs@gmail.com",
    description="Modular quantum computing engine for gyroscopic stabilized systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tnsr-Q/Gyroscopic-stabilized-",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.12.0"],
        "recording": ["pandas>=1.3.0", "pyarrow>=5.0.0"],
        "optimization": ["cvxpy>=1.1.0"],
    },
    entry_points={
        "console_scripts": [
            "gyro-engine=apps.engine.main:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    include_package_data=True,
    zip_safe=False,
)