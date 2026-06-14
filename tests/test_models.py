"""
Unit tests for fraud detection models and components.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Import modules (adjust path as needed)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import DataProcessor
from src.model_training import FraudModelTrainer
from src.anomaly_detection import AnomalyDetector


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
        weights=[0.95, 0.05]
    )
    return X, y


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'amount': np.random.exponential(100, n_samples),
        'customer_id': np.random.randint(100, 200, n_samples),
        'merchant_id': np.random.randint(10, 50, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)


class TestDataProcessor:
    """Tests for DataProcessor class."""
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor(test_size=0.2, random_state=42)
        assert processor.test_size == 0.2
        assert processor.random_state == 42
    
    def test_handle_missing_values(self, sample_dataframe):
        """Test missing value handling."""
        df = sample_dataframe.copy()
        df.loc[0:10, 'amount'] = np.nan
        
        processor = DataProcessor()
        df_clean = processor.handle_missing_values(df, strategy='drop')
        
        assert df_clean['amount'].isna().sum() == 0
    
    def test_remove_duplicates(self, sample_dataframe):
        """Test duplicate removal."""
        df = pd.concat([sample_dataframe, sample_dataframe.iloc[:10]], ignore_index=True)
        
        processor = DataProcessor()
        df_clean = processor.remove_duplicates(df)
        
        assert len(df_clean) < len(df)


class TestFraudModelTrainer:
    """Tests for FraudModelTrainer class."""
    
    def test_model_creation(self):
        """Test model creation."""
        trainer = FraudModelTrainer()
        model = trainer.create_model()
        
        assert model is not None
        assert trainer.model is not None
    
    def test_model_training(self, sample_data):
        """Test model training."""
        X, y = sample_data
        X_train = pd.DataFrame(X[:800])
        y_train = y[:800]
        
        trainer = FraudModelTrainer()
        trainer.create_model(n_estimators=10)
        trainer.train(X_train, y_train)
        
        assert trainer.model is not None
    
    def test_predictions(self, sample_data):
        """Test model predictions."""
        X, y = sample_data
        X_train = pd.DataFrame(X[:800])
        y_train = y[:800]
        X_test = pd.DataFrame(X[800:])
        
        trainer = FraudModelTrainer()
        trainer.create_model(n_estimators=10)
        trainer.train(X_train, y_train)
        predictions = trainer.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""
    
    def test_initialization(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector(contamination=0.05)
        assert detector.contamination == 0.05
        assert detector.model is None
    
    def test_model_fitting(self, sample_data):
        """Test model fitting."""
        X, _ = sample_data
        
        detector = AnomalyDetector(contamination=0.05)
        detector.fit(X)
        
        assert detector.model is not None
    
    def test_predictions(self, sample_data):
        """Test anomaly predictions."""
        X, _ = sample_data
        X_train = X[:800]
        X_test = X[800:]
        
        detector = AnomalyDetector(contamination=0.05)
        detector.fit(X_train)
        predictions = detector.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [-1, 1] for pred in predictions)
    
    def test_anomaly_scores(self, sample_data):
        """Test anomaly score computation."""
        X, _ = sample_data
        
        detector = AnomalyDetector(contamination=0.05)
        detector.fit(X)
        scores = detector.score_samples(X)
        
        assert len(scores) == len(X)
        assert all(isinstance(score, (int, float)) for score in scores)
    
    def test_get_anomaly_percentage(self, sample_data):
        """Test anomaly percentage calculation."""
        X, _ = sample_data
        
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(X)
        percentage = detector.get_anomaly_percentage(X)
        
        assert isinstance(percentage, float)
        assert 0 <= percentage <= 100


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self, sample_dataframe):
        """Test complete ML pipeline."""
        df = sample_dataframe.copy()
        
        # Data processing
        processor = DataProcessor(test_size=0.2, random_state=42)
        df_clean = processor.handle_missing_values(df)
        df_clean = processor.remove_duplicates(df_clean)
        
        X = df_clean.drop(columns=['is_fraud'])
        y = df_clean['is_fraud']
        
        X_train, X_test, y_train, y_test = processor.split_data(X, y)
        
        # Model training
        trainer = FraudModelTrainer()
        trainer.create_model(n_estimators=10)
        trainer.train(X_train, y_train)
        
        # Make predictions
        predictions = trainer.predict(X_test)
        assert len(predictions) == len(X_test)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
