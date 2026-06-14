"""
Model Training Module
Trains XGBoost classifier for supervised fraud detection.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """
    XGBoost model training for fraud detection.
    """

    def __init__(self, random_state=42, scale_pos_weight=None):
        """
        Initialize FraudModelTrainer.
        
        Args:
            random_state (int): Random seed
            scale_pos_weight (float): Weight for positive class
        """
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.model = None
        self.metrics = {}
        self.feature_importance = None

    def create_model(self, max_depth=6, learning_rate=0.1, n_estimators=100, scale_pos_weight=None):
        """
        Create XGBoost model.
        
        Args:
            max_depth (int): Maximum depth of trees
            learning_rate (float): Learning rate
            n_estimators (int): Number of boosting rounds
            scale_pos_weight (float): Weight for positive class (handles imbalance)
            
        Returns:
            XGBClassifier: Initialized model
        """
        if scale_pos_weight is None:
            scale_pos_weight = self.scale_pos_weight
        
        self.model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=-1
        )
        
        logger.info("XGBoost model created")
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Train XGBoost model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation labels
            early_stopping_rounds (int): Early stopping patience
            
        Returns:
            XGBClassifier: Trained model
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        logger.info("Model training completed")
        return self.model

    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Binary predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get probability predictions.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Performance metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Model Evaluation - Accuracy: {self.metrics['accuracy']:.4f}, "
                   f"F1: {self.metrics['f1_score']:.4f}, ROC-AUC: {self.metrics['roc_auc']:.4f}")
        
        return self.metrics

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from trained model.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df.head(top_n)
        logger.info(f"Top {top_n} features extracted")
        
        return self.feature_importance

    def get_model_metrics(self):
        """
        Get all computed metrics.
        
        Returns:
            dict: Model metrics
        """
        return self.metrics

    def get_model_info(self):
        """
        Get model information.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "Not trained"}
        
        params = self.model.get_params()
        
        info = {
            "model_type": "XGBoost",
            "max_depth": params.get('max_depth'),
            "learning_rate": params.get('learning_rate'),
            "n_estimators": params.get('n_estimators'),
            "status": "Trained",
            "metrics": self.metrics
        }
        
        return info

    def save_model(self, filepath):
        """
        Save trained model to file.
        
        Args:
            filepath (str): Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load trained model from file.
        
        Args:
            filepath (str): Path to model file
        """
        self.model = XGBClassifier()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
