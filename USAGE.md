# Usage Guide

## Table of Contents

1. [Data Processing](#data-processing)
2. [Feature Engineering](#feature-engineering)
3. [Model Training](#model-training)
4. [Anomaly Detection](#anomaly-detection)
5. [Model Explainability](#model-explainability)
6. [Real-time Predictions](#real-time-predictions)
7. [Dashboard](#dashboard)

## Data Processing

### Loading Data

```python
from src.data_processing import DataProcessor

processor = DataProcessor(test_size=0.2, random_state=42)
df = processor.load_data('data/raw/transactions.csv')
print(f"Data shape: {df.shape}")
```

### Handling Missing Values

```python
# Drop missing values
df_clean = processor.handle_missing_values(df, strategy='drop')

# Or fill with mean
df_clean = processor.handle_missing_values(df, strategy='mean')
```

### Removing Duplicates

```python
df_unique = processor.remove_duplicates(df)
print(f"Duplicate rows removed: {len(df) - len(df_unique)}")
```

### Removing Outliers

```python
df_clean = processor.remove_outliers(
    df, 
    columns=['amount', 'customer_id'],
    threshold=3  # Z-score threshold
)
```

### Complete Pipeline

```python
X_train, X_test, y_train, y_test = processor.preprocess_pipeline(
    df=transactions_df,
    target_col='is_fraud',
    categorical_cols=['merchant_id', 'location']
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

## Feature Engineering

### Creating Temporal Features

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df = engineer.create_temporal_features(df, timestamp_col='timestamp')

# Created features: hour, day_of_week, day_of_month, month, is_weekend, is_night
```

### Creating Amount Features

```python
df = engineer.create_amount_features(df, amount_col='amount')

# Created features: log_amount, amount_squared, high_amount, low_amount
```

### Creating Customer Features

```python
df = engineer.create_customer_features(df, customer_col='customer_id')

# Created features: total_spent, avg_amount, transaction_count, amount_deviation
```

### Complete Feature Engineering

```python
df_engineered = engineer.feature_engineering_pipeline(df)
features_created = engineer.get_feature_names()

print(f"Total features created: {len(features_created)}")
print(f"Features: {features_created}")
```

## Model Training

### Creating Model

```python
from src.model_training import FraudModelTrainer

trainer = FraudModelTrainer(random_state=42)
model = trainer.create_model(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
```

### Training Model

```python
trainer.train(
    X_train, y_train,
    X_val=X_test,
    y_val=y_test,
    early_stopping_rounds=10
)
```

### Making Predictions

```python
# Binary predictions
predictions = trainer.predict(X_test)
print(f"Predictions: {predictions}")

# Probability predictions
proba = trainer.predict_proba(X_test)
print(f"Fraud probability: {proba[:, 1]}")
```

### Evaluating Model

```python
metrics = trainer.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
```

### Getting Feature Importance

```python
importance_df = trainer.get_feature_importance(top_n=20)
print(importance_df)

# Visualization
import matplotlib.pyplot as plt
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importance')
plt.show()
```

### Saving/Loading Model

```python
# Save model
trainer.save_model('models/xgboost_model.pkl')

# Load model
trainer.load_model('models/xgboost_model.pkl')
```

## Anomaly Detection

### Creating Detector

```python
from src.anomaly_detection import AnomalyDetector

detector = AnomalyDetector(
    contamination=0.05,  # Expected fraud rate
    n_estimators=100
)
```

### Fitting Model

```python
detector.fit(X_train)
```

### Making Predictions

```python
# -1 = anomaly (fraud), 1 = normal
predictions = detector.predict(X_test)
```

### Getting Anomaly Scores

```python
scores = detector.score_samples(X_test)
print(f"Anomaly scores (lower = more anomalous): {scores}")
```

### Getting Statistics

```python
anomaly_percentage = detector.get_anomaly_percentage(X_test)
print(f"Detected anomalies: {anomaly_percentage:.2f}%")
```

## Model Explainability

### Creating Explainer

```python
from src.explainability import SHAPExplainer

explainer = SHAPExplainer(model, X_train, model_type='xgboost')
```

### Computing SHAP Values

```python
shap_values = explainer.compute_shap_values(X_test)
```

### Feature Importance

```python
importance_df = explainer.get_feature_importance(X_test, top_n=20)
print(importance_df)
```

### Explaining Instance

```python
explanation = explainer.explain_instance(X_test, instance_idx=0)

print(f"Base value: {explanation['base_value']}")
print(f"Prediction: {explanation['prediction']}")
print(f"Top contributing features:")
for contrib in explanation['feature_contributions']:
    print(f"  {contrib['feature']}: {contrib['shap_value']:.4f}")
```

## Real-time Predictions

### Creating Inference Engine

```python
from src.inference import FraudDetectionInference

inference = FraudDetectionInference(
    xgb_model=xgb_model,
    iso_forest_model=iso_forest,
    scaler=scaler
)
```

### Single Transaction Prediction

```python
transaction = {
    'amount': 150.00,
    'customer_id': 1001,
    'merchant_id': 501,
    'hour': 14,
    'location': 'USA'
}

result = inference.predict_transaction(transaction)

print(f"Is Fraud: {result['is_fraud']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Risk Level: {result['risk_level']}")
```

### Batch Predictions

```python
transactions_df = pd.read_csv('transactions.csv')
results = inference.predict_batch(transactions_df)

for result in results[:5]:
    print(result)
```

### Ensemble Predictions

```python
result = inference.predict_ensemble(
    X=transactions_df,
    xgb_weight=0.6,
    iso_weight=0.4,
    threshold=0.5
)

print(f"Ensemble Score: {result['ensemble_score']}")
print(f"Ensemble Prediction: {result['ensemble_prediction']}")
```

## Dashboard

### Running Dashboard

```bash
streamlit run dashboard/app.py
```

### Dashboard Pages

#### 1. Dashboard Page
- View real-time fraud metrics
- Transaction amount distribution
- Fraud by location visualization
- Fraud probability trends

#### 2. Transactions Page
- Filter transactions by type
- Search by amount range
- View fraud probability scores
- Export transaction data

#### 3. Model Performance Page
- View model accuracy metrics
- Confusion matrix
- ROC curve visualization
- Feature importance chart

#### 4. Settings Page
- Adjust model hyperparameters
- Configure thresholds
- Manage detection rules
- Save configurations

## Advanced Usage

### Custom Feature Engineering

```python
def custom_features(df):
    # Your custom feature logic
    df['custom_feature'] = df['amount'] * df['hour']
    return df

df = engineer.feature_engineering_pipeline(df)
df = custom_features(df)
```

### Model Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
```

## Performance Optimization

### Parallel Processing

```python
import multiprocessing

n_jobs = multiprocessing.cpu_count()
trainer = FraudModelTrainer()
trainer.create_model(n_jobs=n_jobs)
```

### Model Compression

```python
import joblib

# Save compressed model
joblib.dump(model, 'model.pkl', compress=3)

# Load model
model = joblib.load('model.pkl')
```
