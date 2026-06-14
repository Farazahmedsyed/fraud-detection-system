"""
Feature Engineering Module
Creates and transforms features for fraud detection models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for transaction fraud detection.
    """

    def __init__(self):
        """Initialize FeatureEngineer."""
        self.feature_list = []

    def create_temporal_features(self, df, timestamp_col='timestamp'):
        """
        Create temporal features from timestamp.
        
        Args:
            df (pd.DataFrame): Input data
            timestamp_col (str): Name of timestamp column
            
        Returns:
            pd.DataFrame: Data with temporal features
        """
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(range(22, 24)) | df['hour'].isin(range(0, 6)).astype(int)
        
        self.feature_list.extend(['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_night'])
        logger.info("Temporal features created")
        
        return df

    def create_amount_features(self, df, amount_col='amount'):
        """
        Create features based on transaction amount.
        
        Args:
            df (pd.DataFrame): Input data
            amount_col (str): Name of amount column
            
        Returns:
            pd.DataFrame: Data with amount features
        """
        df['log_amount'] = np.log1p(df[amount_col])
        df['amount_squared'] = df[amount_col] ** 2
        df['high_amount'] = (df[amount_col] > df[amount_col].quantile(0.75)).astype(int)
        df['low_amount'] = (df[amount_col] < df[amount_col].quantile(0.25)).astype(int)
        
        self.feature_list.extend(['log_amount', 'amount_squared', 'high_amount', 'low_amount'])
        logger.info("Amount features created")
        
        return df

    def create_customer_features(self, df, customer_col='customer_id'):
        """
        Create aggregate features per customer.
        
        Args:
            df (pd.DataFrame): Input data
            customer_col (str): Name of customer ID column
            
        Returns:
            pd.DataFrame: Data with customer features
        """
        customer_stats = df.groupby(customer_col).agg({
            'amount': ['sum', 'mean', 'std', 'count'],
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'total_spent', 'avg_amount', 'std_amount', 'transaction_count']
        
        df = df.merge(customer_stats, left_on=customer_col, right_on='customer_id', how='left')
        
        df['std_amount'] = df['std_amount'].fillna(0)
        df['amount_deviation'] = np.abs(df['amount'] - df['avg_amount']) / (df['std_amount'] + 1)
        
        self.feature_list.extend(['total_spent', 'avg_amount', 'transaction_count', 'amount_deviation'])
        logger.info("Customer aggregation features created")
        
        return df

    def create_merchant_features(self, df, merchant_col='merchant_id'):
        """
        Create aggregate features per merchant.
        
        Args:
            df (pd.DataFrame): Input data
            merchant_col (str): Name of merchant ID column
            
        Returns:
            pd.DataFrame: Data with merchant features
        """
        merchant_stats = df.groupby(merchant_col).agg({
            'amount': ['mean', 'std', 'count'],
        }).reset_index()
        
        merchant_stats.columns = ['merchant_id', 'merchant_avg_amount', 'merchant_std_amount', 'merchant_transaction_count']
        
        df = df.merge(merchant_stats, left_on=merchant_col, right_on='merchant_id', how='left')
        
        self.feature_list.extend(['merchant_avg_amount', 'merchant_transaction_count'])
        logger.info("Merchant aggregation features created")
        
        return df

    def create_velocity_features(self, df, customer_col='customer_id', timestamp_col='timestamp', window_days=30):
        """
        Create velocity features (transaction frequency over time windows).
        
        Args:
            df (pd.DataFrame): Input data
            customer_col (str): Name of customer ID column
            timestamp_col (str): Name of timestamp column
            window_days (int): Window size in days
            
        Returns:
            pd.DataFrame: Data with velocity features
        """
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Transactions in last N days
        df['transactions_30d'] = df.groupby(customer_col)[timestamp_col].transform(
            lambda x: (x - x.max()).dt.days >= -window_days
        ).astype(int)
        
        df['transactions_7d'] = df.groupby(customer_col)[timestamp_col].transform(
            lambda x: (x - x.max()).dt.days >= -7
        ).astype(int)
        
        self.feature_list.extend(['transactions_30d', 'transactions_7d'])
        logger.info("Velocity features created")
        
        return df

    def create_location_features(self, df, location_col='location'):
        """
        Create features based on transaction location.
        
        Args:
            df (pd.DataFrame): Input data
            location_col (str): Name of location column
            
        Returns:
            pd.DataFrame: Data with location features
        """
        location_counts = df[location_col].value_counts()
        df['location_frequency'] = df[location_col].map(location_counts)
        df['is_rare_location'] = (df['location_frequency'] < df['location_frequency'].quantile(0.1)).astype(int)
        
        self.feature_list.extend(['location_frequency', 'is_rare_location'])
        logger.info("Location features created")
        
        return df

    def create_interaction_features(self, df):
        """
        Create interaction features between existing features.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with interaction features
        """
        if 'amount' in df.columns and 'transaction_count' in df.columns:
            df['amount_per_transaction'] = df['amount'] / (df['transaction_count'] + 1)
            self.feature_list.append('amount_per_transaction')
        
        if 'amount' in df.columns and 'hour' in df.columns:
            df['night_high_amount'] = (df['is_night'] * df['high_amount']).astype(int)
            self.feature_list.append('night_high_amount')
        
        logger.info("Interaction features created")
        return df

    def feature_engineering_pipeline(self, df, categorical_cols=None):
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Raw data
            categorical_cols (list, optional): Categorical columns to encode
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        # Temporal features
        if 'timestamp' in df.columns:
            df = self.create_temporal_features(df)
        
        # Amount features
        if 'amount' in df.columns:
            df = self.create_amount_features(df)
        
        # Customer features
        if 'customer_id' in df.columns:
            df = self.create_customer_features(df)
            df = self.create_velocity_features(df)
        
        # Merchant features
        if 'merchant_id' in df.columns:
            df = self.create_merchant_features(df)
        
        # Location features
        if 'location' in df.columns:
            df = self.create_location_features(df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering completed. Total features created: {len(self.feature_list)}")
        return df

    def get_feature_names(self):
        """Get list of created feature names."""
        return self.feature_list
