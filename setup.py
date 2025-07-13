"""Setup configuration for PriceFly airline pricing simulation platform."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pricefly",
    version="1.0.0",
    author="PriceFly Development Team",
    author_email="contact@pricefly.ai",
    description="Comprehensive airline pricing simulation platform with advanced revenue optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pricefly/pricefly",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.6.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.931",
            "isort>=5.9.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
            "torch-gpu>=1.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pricefly=pricefly.main:main",
            "pricefly-generate=pricefly.data.synthetic_data:main",
            "pricefly-simulate=pricefly.simulation.engine:main",
            "pricefly-analyze=pricefly.analytics.insights:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pricefly": [
            "data/templates/*.json",
            "data/configs/*.yaml",
            "simulation/scenarios/*.json",
            "analytics/templates/*.html",
            "static/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "airline",
        "pricing",
        "simulation",
        "revenue management",
        "optimization",
        "machine learning",
        "aviation",
        "economics",
        "forecasting",
        "analytics",
    ],
    project_urls={
        "Bug Reports": "https://github.com/pricefly/pricefly/issues",
        "Source": "https://github.com/pricefly/pricefly",
        "Documentation": "https://pricefly.readthedocs.io/",
    },
)