"""
Setup script for Fraud Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fraud-detection-system",
    version="1.0.0",
    author="Faraz Ahmed Syed",
    author_email="your-email@example.com",
    description="A machine learning system for detecting financial fraud using XGBoost and Isolation Forest",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Farazahmedsyed/fraud-detection-system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "xgboost>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "shap>=0.42.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "python-dotenv>=1.0.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "flask>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
