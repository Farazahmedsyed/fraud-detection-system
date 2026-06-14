# API Documentation

## Data Processing API

### DataProcessor Class

```python
from src.data_processing import DataProcessor

processor = DataProcessor(test_size=0.2, random_state=42)
```

#### Methods

- `load_data(filepath)` - Load CSV data
- `handle_missing_values(df, strategy='drop')` - Handle missing values
- `remove_duplicates(df)` - Remove duplicate rows
- `remove_outliers(df, columns, threshold=3)` - Remove outliers
- `encode_categorical(df, categorical_cols)` - One-hot encode
- `scale_features(X_train, X_test=None)` - Scale features
- `split_data(X, y)` - Train-test split
- `preprocess_pipeline(df, target_col, categorical_cols=None)` - Complete pipeline

## Feature Engineering API

### FeatureEngineer Class

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
```

#### Methods

- `create_temporal_features(df, timestamp_col='timestamp')`
- `create_amount_features(df, amount_col='amount')`
- `create_customer_features(df, customer_col='customer_id')`
- `create_merchant_features(df, merchant_col='merchant_id')`
- `create_velocity_features(df, customer_col, timestamp_col, window_days=30)`
- `create_location_features(df, location_col='location')`
- `create_interaction_features(df)`
- `feature_engineering_pipeline(df, categorical_cols=None)`
- `get_feature_names()` - Get list of created features

## Model Training API

### FraudModelTrainer Class

```python
from src.model_training import FraudModelTrainer

trainer = FraudModelTrainer(random_state=42)
```

#### Methods

- `create_model(max_depth=6, learning_rate=0.1, n_estimators=100, scale_pos_weight=None)`
- `train(X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10)`
- `predict(X)` - Get predictions
- `predict_proba(X)` - Get probability predictions
- `evaluate(X_test, y_test)` - Get evaluation metrics
- `get_feature_importance(top_n=20)` - Feature importance
- `save_model(filepath)` - Save model
- `load_model(filepath)` - Load model
- `get_model_info()` - Get model information

## Anomaly Detection API

### AnomalyDetector Class

```python
from src.anomaly_detection import AnomalyDetector

detector = AnomalyDetector(contamination=0.05)
```

#### Methods

- `fit(X)` - Fit model
- `predict(X)` - Get predictions (-1 or 1)
- `score_samples(X)` - Get anomaly scores
- `predict_with_scores(X)` - Get predictions and scores
- `get_anomaly_percentage(X)` - Percentage of anomalies
- `set_contamination(contamination)` - Update contamination
- `get_model_info()` - Get model information

## Explainability API

### SHAPExplainer Class

```python
from src.explainability import SHAPExplainer

explainer = SHAPExplainer(model, X_train, model_type='xgboost')
```

#### Methods

- `compute_shap_values(X)` - Compute SHAP values
- `get_feature_importance(X, top_n=20)` - Feature importance
- `explain_instance(X, instance_idx)` - Explain single instance
- `get_decision_plot_data(X, max_display=20)` - Decision plot data
- `get_summary_plot_data(X)` - Summary plot data
- `get_force_plot_data(X, instance_idx)` - Force plot data

## Inference API

### FraudDetectionInference Class

```python
from src.inference import FraudDetectionInference

inference = FraudDetectionInference(xgb_model, iso_forest_model, scaler)
```

#### Methods

- `load_models(xgb_path, iso_forest_path, scaler_path)` - Load models
- `preprocess_transaction(transaction)` - Preprocess single transaction
- `predict_xgboost(X, return_proba=True)` - XGBoost prediction
- `predict_isolation_forest(X)` - Isolation Forest prediction
- `get_anomaly_scores(X)` - Anomaly scores
- `predict_ensemble(X, xgb_weight=0.6, iso_weight=0.4, threshold=0.5)` - Ensemble prediction
- `predict_transaction(transaction, use_ensemble=True)` - Predict single transaction
- `predict_batch(transactions)` - Batch predictions
- `get_prediction_statistics()` - Prediction statistics
- `clear_log()` - Clear prediction log

## Return Types

### Prediction Result

```python
{
    'prediction': 0 or 1,
    'confidence': float (0-1),
    'is_fraud': bool,
    'risk_level': 'low'|'medium'|'high'|'critical'
}
```

### Ensemble Result

```python
{
    'xgb_probability': np.ndarray,
    'iso_anomaly_score': np.ndarray,
    'iso_fraud_probability': np.ndarray,
    'ensemble_score': np.ndarray,
    'ensemble_prediction': np.ndarray,
    'is_fraud': np.ndarray
}
```

### Evaluation Metrics

```python
{
    'accuracy': float,
    'f1_score': float,
    'roc_auc': float,
    'confusion_matrix': list,
    'classification_report': dict
}
```

## Error Handling

All API methods include proper error handling:

```python
try:
    predictions = trainer.predict(X_test)
except ValueError as e:
    print(f"Error: {e}")
```
