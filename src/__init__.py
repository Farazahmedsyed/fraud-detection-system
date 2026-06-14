"""
Financial Fraud Detection System
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Faraz Ahmed Syed"

from . import data_processing
from . import feature_engineering
from . import model_training
from . import anomaly_detection
from . import explainability
from . import inference

__all__ = [
    "data_processing",
    "feature_engineering",
    "model_training",
    "anomaly_detection",
    "explainability",
    "inference",
]
