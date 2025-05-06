"""
Unit tests for feature engineering functionality.

Tests cover:
- Global feature extraction
- Customer-specific feature extraction
- Feature distributions for transaction simulator
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.features.global_features import (
    add_global_features, add_time_features, add_amount_features,
    add_card_features, add_velocity_features, add_email_device_features
)
from src.features.customer_features import (
    add_customer_features, add_customer_transaction_stats,
    add_customer_behavior_patterns, add_customer_consistency_measures,
    add_customer_deviation_features
)
from src.bootstrap import set_project_root

# Set project root to resolve imports correctly
set_project_root()

class TestGlobalFeatures(unittest.TestCase):
    """Test cases for global feature engineering."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing global features
        self.test_df = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'TransactionDT': [100000, 200000, 300000, 400000, 500000],
            'TransactionAmt': [10.5, 20.75, 100.0, 45.5, 1000.75],
            'card1': [1001, 1002, 1001, 1003, 1002],
            'card4': ['visa', 'mastercard', 'visa', 'visa', 'mastercard'],
            'P_emaildomain': ['gmail.com', 'yahoo.com', 'gmail.com', 'outlook.com', 'hotmail.com'],
            'DeviceType': ['mobile', 'desktop', 'mobile', 'mobile', 'desktop'],
            'isFraud': [0, 0, 1, 0, 1]
        })
    
    def test_add_time_features(self):
        """Test addition of time-based features."""
        result_df = add_time_features(self.test_df)
        
        # Check that time features were added
        self.assertIn('hour', result_df.columns)
        self.assertIn('day', result_df.columns)
        self.assertIn('dow', result_df.columns)
        self.assertIn('is_weekend', result_df.columns)
        self.assertIn('is_business_hours', result_df.columns)
        
        # Verify hour calculations
        # TransactionDT 100000 is approximately 27.78 hours or 1 day + 3.78 hours
        self.assertAlmostEqual(result_df.loc[0, 'hour'], 3, delta=1)
    
    def test_add_amount_features(self):
        """Test addition of amount-related features."""
        result_df = add_amount_features(self.test_df)
        
        # Check amount features
        self.assertIn('TransactionAmt_log', result_df.columns)
        self.assertIn('TransactionAmt_bin', result_df.columns)
        self.assertIn('is_round_amount', result_df.columns)
        
        # Verify log transformation
        self.assertAlmostEqual(
            result_df.loc[0, 'TransactionAmt_log'],
            np.log1p(self.test_df.loc[0, 'TransactionAmt']),
            places=4
        )
        
        # Check round amount flag
        # The amount 100.0 should be flagged as round
        self.assertEqual(result_df.loc[2, 'is_round_amount'], 1)
    
    def test_add_card_features(self):
        """Test addition of card-related features."""
        result_df = add_card_features(self.test_df)
        
        # Check card features
        self.assertIn('card1_count', result_df.columns)
        
        # Verify card1 count
        # Card1 1001 appears twice
        self.assertEqual(result_df.loc[0, 'card1_count'], 2)
        self.assertEqual(result_df.loc[2, 'card1_count'], 2)
        
        # Card1 1002 appears twice
        self.assertEqual(result_df.loc[1, 'card1_count'], 2)
        self.assertEqual(result_df.loc[4, 'card1_count'], 2)
        
        # Card1 1003 appears once
        self.assertEqual(result_df.loc[3, 'card1_count'], 1)
    
    def test_add_velocity_features(self):
        """Test addition of velocity features."""
        # Use only 1 hour window for test
        result_df = add_velocity_features(self.test_df, windows=[1])
        
        # Check velocity features
        self.assertIn('velocity_1h', result_df.columns)
        
        # Verify velocity calculations
        # For the first transaction of each card, velocity should be 0
        card_first_txns = self.test_df.sort_values(['card1', 'TransactionDT']).drop_duplicates('card1')
        for idx, row in card_first_txns.iterrows():
            self.assertEqual(result_df.loc[idx, 'velocity_1h'], 0)
    
    def test_add_email_device_features(self):
        """Test addition of email and device features."""
        result_df = add_email_device_features(self.test_df)
        
        # Check email features
        self.assertIn('P_emaildomain_root', result_df.columns)
        self.assertIn('P_emaildomain_count', result_df.columns)
        
        # Check device features
        if 'DeviceType' in self.test_df.columns:
            self.assertIn('device_count', result_df.columns)
            
            # Verify device count
            # 'mobile' appears 3 times, 'desktop' appears 2 times
            mobile_indices = self.test_df[self.test_df['DeviceType'] == 'mobile'].index
            desktop_indices = self.test_df[self.test_df['DeviceType'] == 'desktop'].index
            
            for idx in mobile_indices:
                self.assertEqual(result_df.loc[idx, 'device_count'], 3)
            
            for idx in desktop_indices:
                self.assertEqual(result_df.loc[idx, 'device_count'], 2)
    
    def test_add_global_features(self):
        """Test the full global feature extraction pipeline."""
        result_df = add_global_features(self.test_df)
        
        # Check that preprocessing completed without errors
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # Check that original columns are preserved
        for col in self.test_df.columns:
            self.assertIn(col, result_df.columns)
        
        # Check that new features were added
        self.assertTrue(len(result_df.columns) > len(self.test_df.columns))
        
        # Check specific feature groups
        self.assertIn('hour', result_df.columns)  # Time features
        self.assertIn('TransactionAmt_log', result_df.columns)  # Amount features
        self.assertIn('card1_count', result_df.columns)  # Card features

class TestCustomerFeatures(unittest.TestCase):
    """Test cases for customer-specific feature engineering."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing customer features
        # Ensure multiple transactions per customer with different timestamps
        self.test_df = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5, 6, 7, 8],
            'TransactionDT': [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000],
            'TransactionAmt': [10.5, 20.75, 100.0, 45.5, 1000.75, 15.0, 50.0, 500.0],
            'card1': [1001, 1001, 1002, 1002, 1002, 1003, 1003, 1001],
            'card4': ['visa', 'visa', 'mastercard', 'mastercard', 'mastercard', 'visa', 'visa', 'visa'],
            'hour': [3, 15, 6, 18, 9, 12, 23, 7],
            'dow': [5, 6, 0, 1, 2, 3, 4, 5],
            'P_emaildomain': ['gmail.com', 'gmail.com', 'yahoo.com', 'yahoo.com', 'hotmail.com', 'gmail.com', 'hotmail.com', 'gmail.com'],
            'DeviceType': ['mobile', 'mobile', 'desktop', 'desktop', 'desktop', 'mobile', 'tablet', 'mobile'],
            'isFraud': [0, 0, 0, 0, 1, 0, 0, 1]
        })
    
    def test_add_customer_transaction_stats(self):
        """Test addition of customer transaction statistics."""
        result_df = add_customer_transaction_stats(self.test_df)
        
        # Check transaction stats features
        self.assertIn('cust_txn_count', result_df.columns)
        self.assertIn('cust_amt_mean', result_df.columns)
        self.assertIn('cust_amt_std', result_df.columns)
        
        # Verify transaction counts
        # Card1 1001 has 3 transactions
        self.assertEqual(result_df.loc[0, 'cust_txn_count'], 3)
        self.assertEqual(result_df.loc[1, 'cust_txn_count'], 3)
        self.assertEqual(result_df.loc[7, 'cust_txn_count'], 3)
        
        # Card1 1002 has 3 transactions
        self.assertEqual(result_df.loc[2, 'cust_txn_count'], 3)
        self.assertEqual(result_df.loc[3, 'cust_txn_count'], 3)
        self.assertEqual(result_df.loc[4, 'cust_txn_count'], 3)
        
        # Card1 1003 has 2 transactions
        self.assertEqual(result_df.loc[5, 'cust_txn_count'], 2)
        self.assertEqual(result_df.loc[6, 'cust_txn_count'], 2)
        
        # Verify amount statistics
        # Card1 1001 amounts: [10.5, 20.75, 500.0]
        card1001_mean = np.mean([10.5, 20.75, 500.0])
        card1001_std = np.std([10.5, 20.75, 500.0])
        
        self.assertAlmostEqual(result_df.loc[0, 'cust_amt_mean'], card1001_mean, places=2)
        self.assertAlmostEqual(result_df.loc[0, 'cust_amt_std'], card1001_std, places=2)
    
    def test_add_customer_behavior_patterns(self):
        """Test addition of customer behavior pattern features."""
        result_df = add_customer_behavior_patterns(self.test_df)
        
        # Check behavior pattern features
        self.assertIn('cust_most_common_hour', result_df.columns)
        
        if 'P_emaildomain' in self.test_df.columns:
            self.assertIn('cust_primary_email', result_df.columns)
        
        if 'DeviceType' in self.test_df.columns:
            self.assertIn('cust_primary_device', result_df.columns)
        
        # Verify hour patterns
        # Card1 1001 hours: [3, 15, 7] - no clear mode, should pick the first occurrence
        # Card1 1002 hours: [6, 18, 9] - no clear mode, should pick the first occurrence
        # Card1 1003 hours: [12, 23] - no clear mode, should pick the first occurrence
        
        # Verify device patterns
        # Card1 1001 devices: ['mobile', 'mobile', 'mobile'] - 'mobile' is mode
        # Card1 1002 devices: ['desktop', 'desktop', 'desktop'] - 'desktop' is mode
        # Card1 1003 devices: ['mobile', 'tablet'] - no clear mode, should pick the first occurrence
        
        self.assertEqual(result_df.loc[0, 'cust_primary_device'], 'mobile')
        self.assertEqual(result_df.loc[2, 'cust_primary_device'], 'desktop')
    
    def test_add_customer_deviation_features(self):
        """Test addition of customer pattern deviation features."""
        # First sort by card1 and time
        test_df_sorted = self.test_df.sort_values(['card1', 'TransactionDT']).reset_index(drop=True)
        
        result_df = add_customer_deviation_features(test_df_sorted)
        
        # Check deviation features
        self.assertIn('txn_seq_number', result_df.columns)
        self.assertIn('time_since_prev_txn', result_df.columns)
        
        if 'TransactionAmt' in test_df_sorted.columns:
            self.assertIn('amt_diff_from_prev', result_df.columns)
            
            # Verify amount difference calculations
            # Second transaction for each customer should have a non-zero difference
            for card in test_df_sorted['card1'].unique():
                card_indices = test_df_sorted[test_df_sorted['card1'] == card].index
                if len(card_indices) > 1:
                    second_idx = card_indices[1]
                    self.assertNotEqual(result_df.loc[second_idx, 'amt_diff_from_prev'], 0)
    
    def test_add_customer_features(self):
        """Test the full customer feature extraction pipeline."""
        result_df = add_customer_features(self.test_df)
        
        # Check that preprocessing completed without errors
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # Check that original columns are preserved
        for col in self.test_df.columns:
            self.assertIn(col, result_df.columns)
        
        # Check that new features were added
        self.assertTrue(len(result_df.columns) > len(self.test_df.columns))
        
        # Check specific feature groups
        self.assertIn('cust_txn_count', result_df.columns)  # Transaction stats
        self.assertIn('cust_primary_device', result_df.columns)  # Behavior patterns
        self.assertIn('cust_amt_std', result_df.columns)  # Transaction statistics

class TestFeatureDistributions(unittest.TestCase):
    """Test cases for feature distributions generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        
        # Create sample data for feature distributions
        self.test_df = pd.DataFrame({
            'TransactionAmt': [10.5, 20.75, 100.0, 45.5, 1000.75],
            'hour': [3, 15, 6, 18, 9],
            'card4': ['visa', 'mastercard', 'visa', 'visa', 'mastercard'],
            'isFraud': [0, 0, 0, 0, 1]
        })
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_feature_distributions_generation(self):
        """Test generation of feature distributions for the transaction simulator."""
        # Generate feature distributions
        dist = {}
        
        # Only use legitimate transactions
        legit_df = self.test_df[self.test_df['isFraud'] == 0]
        
        # Process numeric features
        for col in legit_df.select_dtypes(include=['number']).columns:
            if col != 'isFraud':
                dist[col] = {
                    'min': float(legit_df[col].min()),
                    'max': float(legit_df[col].max()),
                    'median': float(legit_df[col].median()),
                    'mean': float(legit_df[col].mean())
                }
        
        # Process categorical features
        for col in legit_df.select_dtypes(exclude=['number']).columns:
            if col != 'isFraud':
                val_counts = legit_df[col].value_counts(normalize=True)
                dist[col] = {
                    'values': val_counts.index.tolist(),
                    'frequencies': val_counts.values.tolist()
                }
        
        # Save distribution to file
        dist_path = self.output_dir / 'feature_distributions.json'
        with open(dist_path, 'w') as f:
            json.dump(dist, f)
        
        # Check file creation
        self.assertTrue(dist_path.exists())
        
        # Load and validate
        with open(dist_path, 'r') as f:
            loaded_dist = json.load(f)
        
        # Check numeric features
        self.assertIn('TransactionAmt', loaded_dist)
        self.assertIn('hour', loaded_dist)
        
        # Check numeric feature properties
        for col in ['TransactionAmt', 'hour']:
            self.assertIn('min', loaded_dist[col])
            self.assertIn('max', loaded_dist[col])
            self.assertIn('median', loaded_dist[col])
            self.assertIn('mean', loaded_dist[col])
        
        # Check categorical features
        self.assertIn('card4', loaded_dist)
        
        # Check categorical feature properties
        self.assertIn('values', loaded_dist['card4'])
        self.assertIn('frequencies', loaded_dist['card4'])
        
        # Verify values for 'visa' and 'mastercard'
        self.assertIn('visa', loaded_dist['card4']['values'])
        self.assertIn('mastercard', loaded_dist['card4']['values'])
        
        # Check that frequencies sum to approximately 1.0
        freq_sum = sum(loaded_dist['card4']['frequencies'])
        self.assertAlmostEqual(freq_sum, 1.0, places=5)
    
    def test_feature_distributions_in_pipeline(self):
        """Test feature distributions in the dodo.py pipeline context."""
        # Create a mock version of the pipeline task
        from src.features.global_features import add_global_features
        
        # Add global features
        enriched_df = add_global_features(self.test_df)
        
        # Create feature distributions dictionary
        dist = {}
        
        # Only use legitimate transactions
        legit_df = enriched_df[enriched_df['isFraud'] == 0]
        
        # Process numeric features (limit to a few for test)
        numeric_features = ['TransactionAmt', 'hour']
        for col in numeric_features:
            if col in legit_df.columns:
                dist[col] = {
                    'min': float(legit_df[col].min()),
                    'max': float(legit_df[col].max()),
                    'median': float(legit_df[col].median()),
                    'mean': float(legit_df[col].mean())
                }
        
        # Process categorical features (limit to a few for test)
        categorical_features = ['card4']
        for col in categorical_features:
            if col in legit_df.columns:
                val_counts = legit_df[col].value_counts(normalize=True)
                dist[col] = {
                    'values': val_counts.index.tolist(),
                    'frequencies': val_counts.values.tolist()
                }
        
        # Save distribution to file
        dist_path = self.output_dir / 'feature_distributions.json'
        with open(dist_path, 'w') as f:
            json.dump(dist, f)
        
        # Verify file creation
        self.assertTrue(dist_path.exists())

if __name__ == "__main__":
    unittest.main()