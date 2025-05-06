"""
Unit tests for data loading and processing functions.

Tests cover:
- Downcast functionality for memory optimization
- Data loading with proper types
- Feature extraction functionality
- Preprocessing steps for IEEE-CIS dataset
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import downcast_dtypes, load_data
from src.data.processor import (
    preprocess_data, handle_missing_values, 
    encode_categorical_features, extract_time_features
)
from src.bootstrap import set_project_root

# Set project root to resolve imports correctly
set_project_root()

class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Create sample transaction data
        self.transaction_df = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'TransactionDT': [100000, 200000, 300000, 400000, 500000],
            'TransactionAmt': [10.5, 20.75, 100.0, 45.5, 1000.75],
            'card1': [1001, 1002, 1001, 1003, 1002],
            'card4': ['visa', 'mastercard', 'visa', 'visa', 'mastercard'],
            'P_emaildomain': ['gmail.com', 'yahoo.com', 'gmail.com', None, 'hotmail.com'],
            'isFraud': [0, 0, 1, 0, 1]
        })
        
        # Create sample identity data
        self.identity_df = pd.DataFrame({
            'TransactionID': [1, 2, 4],
            'DeviceType': ['mobile', 'desktop', 'mobile'],
            'DeviceInfo': ['iOS', 'Windows', 'Android'],
            'id_01': [100.0, 200.0, 150.0],
            'id_02': [0.5, 0.7, 0.3]
        })
        
        # Save test data
        self.transaction_df.to_csv(self.data_dir / 'train_transaction.csv', index=False)
        self.identity_df.to_csv(self.data_dir / 'train_identity.csv', index=False)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_downcast_dtypes(self):
        """Test downcast_dtypes function for memory optimization."""
        # Create a test DataFrame with various dtypes
        test_df = pd.DataFrame({
            'int64_col': np.array([1, 2, 3], dtype=np.int64),
            'int32_col': np.array([1, 2, 3], dtype=np.int32),
            'float64_col': np.array([1.0, 2.0, 3.0], dtype=np.float64),
            'small_int': np.array([1, 2, 3], dtype=np.int64),
            'uint_candidate': np.array([10, 20, 30], dtype=np.int64),
            'str_col': ['a', 'b', 'c']
        })
        
        # Apply downcast
        result_df = downcast_dtypes(test_df)
        
        # Check dtypes
        self.assertIn(result_df['int64_col'].dtype.name, ['int8', 'int16', 'int32'])
        self.assertIn(result_df['small_int'].dtype.name, ['int8', 'int16'])
        self.assertIn(result_df['uint_candidate'].dtype.name, ['uint8', 'uint16'])
        self.assertEqual(result_df['float64_col'].dtype, np.dtype('float32'))
        self.assertEqual(result_df['str_col'].dtype, test_df['str_col'].dtype)
        
        # Check values are preserved
        pd.testing.assert_series_equal(test_df['int64_col'], result_df['int64_col'], check_dtype=False)
        pd.testing.assert_series_equal(test_df['float64_col'], result_df['float64_col'], check_dtype=False)
        pd.testing.assert_series_equal(test_df['str_col'], result_df['str_col'])
    
    def test_downcast_csv(self):
        """Test downcast_csv function for file-based downcasting."""
        from src.data.loader import downcast_csv
        
        # Create a temporary CSV file
        test_df = pd.DataFrame({
            'int_col': [100, 200, 300],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        
        input_path = self.data_dir / 'test_original.csv'
        output_path = self.data_dir / 'test_downcast.csv'
        
        # Save test DataFrame
        test_df.to_csv(input_path, index=False)
        
        # Apply downcast_csv
        downcast_csv(str(input_path), str(output_path))
        
        # Read back the downcast CSV
        result_df = pd.read_csv(output_path)
        
        # Check file was created
        self.assertTrue(output_path.exists())
        
        # Check row count matches
        self.assertEqual(len(test_df), len(result_df))
        
        # Check column names match
        self.assertListEqual(list(test_df.columns), list(result_df.columns))
        
        # Check numeric types are downcast
        self.assertNotEqual(test_df['int_col'].dtype, np.int64)
        self.assertNotEqual(test_df['float_col'].dtype, np.float64)
    
    def test_load_data(self):
        """Test load_data function for loading and merging transaction and identity data."""
        # Load data
        df = load_data(self.data_dir, train=True)
        
        # Check DataFrame properties
        self.assertEqual(len(df), 5)  # Should contain all 5 transactions
        
        # Check if identity data was merged
        self.assertIn('DeviceType', df.columns)
        
        # Check that identity data was merged correctly
        # TransactionID 3 and 5 should have NaN for identity columns
        self.assertTrue(pd.isna(df.loc[df['TransactionID'] == 3, 'DeviceType'].iloc[0]))
        self.assertTrue(pd.isna(df.loc[df['TransactionID'] == 5, 'DeviceType'].iloc[0]))
        
        # Check proper dtypes for numeric columns
        self.assertNotEqual(df['TransactionAmt'].dtype, np.float64)

class TestDataProcessor(unittest.TestCase):
    """Test cases for data processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with various issues to test preprocessing
        self.test_df = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'TransactionDT': [100000, 200000, 300000, 400000, 500000],
            'TransactionAmt': [10.5, 20.75, 100.0, None, 1000.75],
            'card1': [1001, 1002, 1001, 1003, 1002],
            'card4': ['visa', 'mastercard', 'visa', None, 'mastercard'],
            'P_emaildomain': ['gmail.com', 'yahoo.com', 'gmail.com', None, 'hotmail.com'],
            'DeviceType': ['mobile', 'desktop', 'mobile', None, 'mobile'],
            'id_01': [100.0, 200.0, None, 150.0, 120.0],
            'isFraud': [0, 0, 1, 0, 1]
        })
    
    def test_handle_missing_values(self):
        """Test handling of missing values in the dataset."""
        result_df = handle_missing_values(self.test_df, target_col='isFraud')
        
        # Check that all missing values were handled
        self.assertEqual(result_df.isna().sum().sum(), 0)
        
        # Check specific imputation strategies
        # Numeric column should use median
        self.assertFalse(pd.isna(result_df.loc[3, 'TransactionAmt']))
        self.assertFalse(pd.isna(result_df.loc[2, 'id_01']))
        
        # Categorical should use 'unknown' or most frequent
        self.assertEqual(result_df.loc[3, 'card4'], 'visa')  # Most frequent
        self.assertEqual(result_df.loc[3, 'P_emaildomain'], 'unknown')
    
    def test_encode_categorical_features(self):
        """Test encoding of categorical features."""
        # Handle missing values first to avoid encoding issues
        clean_df = handle_missing_values(self.test_df, target_col='isFraud')
        
        # Apply encoding
        result_df = encode_categorical_features(clean_df, target_col='isFraud')
        
        # Check binary encoding (card4 has two values, so should be converted to 0/1)
        if 'card4' in result_df.columns:
            self.assertIn(result_df['card4'].dtype.name, ['int64', 'int32', 'int16', 'int8'])
            self.assertEqual(result_df['card4'].nunique(), 2)
        
        # Check email domain encoding (should create new column)
        self.assertIn('P_emaildomain_root', result_df.columns)
        self.assertTrue(result_df['P_emaildomain_root'].dtype.name != 'object')
    
    def test_extract_time_features(self):
        """Test extraction of time features from TransactionDT."""
        result_df = extract_time_features(self.test_df)
        
        # Check that time features were added
        self.assertIn('transaction_hour', result_df.columns)
        self.assertIn('day_of_week', result_df.columns)
        self.assertIn('day_of_month', result_df.columns)
        
        # Check derived boolean features
        self.assertIn('is_weekend', result_df.columns)
        self.assertIn('is_night', result_df.columns)
        
        # Check values (IEEE-CIS dataset starts on 2017-12-01, which was a Friday = day 4)
        # TransactionDT is in seconds, 100000 seconds = ~27.7 hours
        # So first transaction should be on day 5 (Saturday)
        self.assertEqual(result_df.loc[0, 'day_of_week'], 5)  # Saturday
        
        # Check hour extraction
        # 100000 seconds = 27.7 hours = 27 hours + 0.7*60 = 27 hours + 42 minutes
        self.assertAlmostEqual(result_df.loc[0, 'transaction_hour'], 3.7, places=1)  # 27 hours % 24 = 3 hours
    
    def test_preprocess_data(self):
        """Test the full preprocessing pipeline."""
        result_df = preprocess_data(self.test_df)
        
        # Check that preprocessing completed without errors
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # Check for missing values
        self.assertEqual(result_df.isna().sum().sum(), 0)
        
        # Check target column is preserved
        self.assertIn('isFraud', result_df.columns)
        
        # Check categorical encoding
        categoricals = self.test_df.select_dtypes(include=['object']).columns
        for col in categoricals:
            if col != 'isFraud':
                # Either original column is encoded or new columns were created
                if col in result_df.columns:
                    self.assertNotEqual(result_df[col].dtype, 'object')

if __name__ == "__main__":
    unittest.main()