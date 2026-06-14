"""
Data Processing Module
Handles data loading, cleaning, and preprocessing for the fraud detection system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data preprocessing pipeline for fraud detection.
    """

    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize DataProcessor.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.feature_names = None

    def load_data(self, filepath):
        """
        Load transaction data from CSV.
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_missing_values(self, df, strategy='drop'):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input data
            strategy (str): Strategy to handle missing values ('drop' or 'mean')
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        logger.info(f"Missing values handled. Remaining rows: {len(df)}")
        return df

    def remove_duplicates(self, df):
        """
        Remove duplicate transactions.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data without duplicates
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} duplicate rows")
        return df

    def remove_outliers(self, df, columns, threshold=3):
        """
        Remove outliers using Z-score method.
        
        Args:
            df (pd.DataFrame): Input data
            columns (list): Columns to check for outliers
            threshold (float): Z-score threshold
            
        Returns:
            pd.DataFrame: Data without outliers
        """
        from scipy import stats
        
        initial_rows = len(df)
        z_scores = np.abs(stats.zscore(df[columns].select_dtypes(include=[np.number])))
        df = df[(z_scores < threshold).all(axis=1)]
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} outlier rows")
        return df

    def encode_categorical(self, df, categorical_cols):
        """
        Encode categorical variables using one-hot encoding.
        
        Args:
            df (pd.DataFrame): Input data
            categorical_cols (list): List of categorical columns
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
        """
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logger.info(f"Categorical encoding completed. New shape: {df.shape}")
        return df

    def scale_features(self, X_train, X_test=None):
        """
        Scale numerical features using RobustScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: Scaled training and test data
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            logger.info("Features scaled successfully")
            return X_train_scaled, X_test_scaled
        
        logger.info("Training features scaled successfully")
        return X_train_scaled

    def split_data(self, X, y):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def preprocess_pipeline(self, df, target_col, categorical_cols=None):
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Raw data
            target_col (str): Name of target column
            categorical_cols (list, optional): List of categorical columns
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Data cleaning
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Encoding
        if categorical_cols:
            X = self.encode_categorical(X, categorical_cols)
        
        # Remove outliers
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = self.remove_outliers(X, numeric_cols)
        y = y[X.index]
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        self.feature_names = X_train_scaled.columns.tolist()
        logger.info("Preprocessing pipeline completed successfully")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
