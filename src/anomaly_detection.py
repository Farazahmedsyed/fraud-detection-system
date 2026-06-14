"""
Anomaly Detection Module
Implements Isolation Forest for unsupervised fraud detection.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Unsupervised anomaly detection using Isolation Forest.
    """

    def __init__(self, contamination=0.05, random_state=42, n_estimators=100):
        """
        Initialize AnomalyDetector.
        
        Args:
            contamination (float): Expected proportion of outliers
            random_state (int): Random seed for reproducibility
            n_estimators (int): Number of trees in the forest
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = None
        self.threshold = None

    def fit(self, X):
        """
        Fit Isolation Forest model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Training data
            
        Returns:
            self: Fitted anomaly detector
        """
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            n_jobs=-1
        )
        
        self.model.fit(X)
        logger.info("Isolation Forest model fitted successfully")
        
        return self

    def predict(self, X):
        """
        Predict anomalies (-1 for anomalies, 1 for normal).
        
        Args:
            X (np.ndarray or pd.DataFrame): Data to predict
            
        Returns:
            np.ndarray: Predictions (-1 or 1)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model.predict(X)
        logger.info(f"Predictions made on {len(X)} samples")
        
        return predictions

    def score_samples(self, X):
        """
        Get anomaly scores for samples (lower = more anomalous).
        
        Args:
            X (np.ndarray or pd.DataFrame): Data to score
            
        Returns:
            np.ndarray: Anomaly scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scores = self.model.score_samples(X)
        return scores

    def predict_with_scores(self, X):
        """
        Get both predictions and scores.
        
        Args:
            X (np.ndarray or pd.DataFrame): Data to predict
            
        Returns:
            tuple: (predictions, scores)
        """
        predictions = self.predict(X)
        scores = self.score_samples(X)
        
        return predictions, scores

    def get_anomaly_percentage(self, X):
        """
        Calculate percentage of detected anomalies.
        
        Args:
            X (np.ndarray or pd.DataFrame): Data to analyze
            
        Returns:
            float: Percentage of anomalies
        """
        predictions = self.predict(X)
        anomaly_count = np.sum(predictions == -1)
        percentage = (anomaly_count / len(predictions)) * 100
        
        return percentage

    def set_contamination(self, contamination):
        """
        Update contamination parameter and refit model.
        
        Args:
            contamination (float): New contamination rate
        """
        self.contamination = contamination
        if self.model is not None:
            logger.info(f"Contamination updated to {contamination}")

    def get_model_info(self):
        """
        Get model information and parameters.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "Not fitted"}
        
        info = {
            "model_type": "Isolation Forest",
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "status": "Fitted"
        }
        
        return info
