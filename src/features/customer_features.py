"""
Individual customer features module for IEEE-CIS fraud detection.

Implements customer-specific features including geospatial analysis:
- Per-customer transaction statistics
- Behavioral profiles
- Activity patterns
- Time window deviations
- Geospatial patterns and distance analysis
- Customer fingerprinting
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from scipy import stats
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add customer-specific features based on card1 identifiers.
    
    Creates features that capture individual customer behaviors:
    - Transaction frequency and timing
    - Amount patterns
    - Behavioral consistency measures
    - Distance and geospatial patterns
    - Deviation from personal historical patterns
    
    Args:
        df: DataFrame with transaction data including card1
        
    Returns:
        DataFrame with added customer-specific features
    """
    df = df.copy()
    logger.info("Adding customer-specific features...")
    
    if 'card1' not in df.columns:
        logger.warning("card1 column not found, skipping customer features.")
        return df
    
    # Apply all customer feature transformations
    df = add_customer_transaction_stats(df)
    df = add_customer_behavior_patterns(df)
    df = add_customer_geospatial_features(df)
    df = add_customer_consistency_measures(df)
    df = add_customer_deviation_features(df)
    
    logger.info(f"Added customer features. New DataFrame shape: {df.shape}")
    return df

def add_customer_transaction_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic transaction statistics per customer.
    
    Args:
        df: DataFrame with card1 and transaction data
        
    Returns:
        DataFrame with added customer transaction statistics
    """
    logger.info("Adding customer transaction statistics...")
    
    # Group by card1
    grouped = df.groupby('card1')
    
    # Transaction count
    df['cust_txn_count'] = grouped.size().reindex(df['card1']).values
    
    # Transaction amount statistics
    if 'TransactionAmt' in df.columns:
        # Amount statistics
        df['cust_amt_mean'] = grouped['TransactionAmt'].mean().reindex(df['card1']).values
        df['cust_amt_std'] = grouped['TransactionAmt'].std().fillna(0).reindex(df['card1']).values
        df['cust_amt_min'] = grouped['TransactionAmt'].min().reindex(df['card1']).values
        df['cust_amt_max'] = grouped['TransactionAmt'].max().reindex(df['card1']).values
        
        # Amount range
        df['cust_amt_range'] = df['cust_amt_max'] - df['cust_amt_min']
        
        # Coefficient of variation (higher = more variable)
        eps = 1e-6  # To avoid division by zero
        df['cust_amt_cv'] = df['cust_amt_std'] / (df['cust_amt_mean'] + eps)
        
        # Z-score within customer
        df['cust_amt_zscore'] = grouped.apply(
            lambda x: (x['TransactionAmt'] - x['TransactionAmt'].mean()) / (x['TransactionAmt'].std() + eps)
        ).values
        
        # Deviation from customer mean
        df['amt_dev_from_cust_mean'] = df['TransactionAmt'] - df['cust_amt_mean']
        df['amt_dev_ratio'] = df['TransactionAmt'] / (df['cust_amt_mean'] + eps)
    
    # Transaction time statistics
    if 'TransactionDT' in df.columns:
        # First transaction time
        df['cust_first_txn'] = grouped['TransactionDT'].min().reindex(df['card1']).values
        
        # Last transaction time
        df['cust_last_txn'] = grouped['TransactionDT'].max().reindex(df['card1']).values
        
        # Account age in days
        df['cust_account_age_days'] = (df['cust_last_txn'] - df['cust_first_txn']) / (24 * 3600)
        
        # Time since first transaction in days
        df['days_since_first_txn'] = (df['TransactionDT'] - df['cust_first_txn']) / (24 * 3600)
        
        # Time since last transaction in hours
        df['hours_since_last_txn'] = (df['TransactionDT'] - df['cust_last_txn']) / 3600
        
        # Average time between transactions in hours
        df['avg_time_between_txns'] = df['cust_account_age_days'] * 24 / (df['cust_txn_count'] + eps)
    
    logger.info("Customer transaction statistics added.")
    return df

def add_customer_behavior_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features capturing customer behavior patterns.
    
    Args:
        df: DataFrame with card1 and transaction data
        
    Returns:
        DataFrame with added behavior pattern features
    """
    logger.info("Adding customer behavior pattern features...")
    
    # Activity hour patterns
    if all(col in df.columns for col in ['card1', 'hour']):
        # Get distribution of transaction hours for each customer
        hour_profiles = pd.crosstab(
            df['card1'], df['hour'], 
            normalize='index'
        )
        
        # Find most common transaction hour for each customer
        df['cust_most_common_hour'] = df['card1'].map(hour_profiles.idxmax(axis=1))
        
        # Calculate if current transaction is in customer's common hours
        df['is_common_hour'] = (df['hour'] == df['cust_most_common_hour']).astype('uint8')
        
        # Night transaction ratio
        night_hours = list(range(0, 6)) + list(range(22, 24))
        df['cust_night_txn_ratio'] = df['card1'].map(
            hour_profiles[night_hours].sum(axis=1)
        )
    
    # Activity day patterns
    if all(col in df.columns for col in ['card1', 'dow']):
        # Get distribution of transaction days for each customer
        dow_profiles = pd.crosstab(
            df['card1'], df['dow'], 
            normalize='index'
        )
        
        # Find most common transaction day for each customer
        df['cust_most_common_dow'] = df['card1'].map(dow_profiles.idxmax(axis=1))
        
        # Calculate if current transaction is on customer's common day
        df['is_common_dow'] = (df['dow'] == df['cust_most_common_dow']).astype('uint8')
        
        # Weekend transaction ratio
        df['cust_weekend_txn_ratio'] = df['card1'].map(
            dow_profiles[[5, 6]].sum(axis=1)
        )
    
    # Email domain preferences
    if all(col in df.columns for col in ['card1', 'P_emaildomain']):
        # Get most common email domain for each customer
        email_counts = df.groupby(['card1', 'P_emaildomain']).size().reset_index(name='count')
        most_common_email = email_counts.sort_values(['card1', 'count'], ascending=[True, False]) \
                            .drop_duplicates('card1')
        
        df['cust_primary_email'] = df['card1'].map(
            dict(zip(most_common_email['card1'], most_common_email['P_emaildomain']))
        )
        
        # Flag when customer uses uncommon email
        df['is_uncommon_email'] = (df['P_emaildomain'] != df['cust_primary_email']).astype('uint8')
    
    # Device preferences
    if all(col in df.columns for col in ['card1', 'DeviceType']):
        # Get most common device for each customer
        device_counts = df.groupby(['card1', 'DeviceType']).size().reset_index(name='count')
        most_common_device = device_counts.sort_values(['card1', 'count'], ascending=[True, False]) \
                            .drop_duplicates('card1')
        
        df['cust_primary_device'] = df['card1'].map(
            dict(zip(most_common_device['card1'], most_common_device['DeviceType']))
        )
        
        # Flag when customer uses uncommon device
        df['is_uncommon_device'] = (df['DeviceType'] != df['cust_primary_device']).astype('uint8')
    
    logger.info("Customer behavior pattern features added.")
    return df

def add_customer_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add customer-specific geospatial features based on transaction locations.
    
    Includes:
    - Distance patterns (dist1, dist2)
    - Location clusters for each customer
    - Unusual location flags
    - Geospatial velocity features (distance/time)
    
    Args:
        df: DataFrame with card1 and location data
        
    Returns:
        DataFrame with added geospatial features
    """
    logger.info("Adding customer geospatial features...")
    
    # Check if geospatial columns exist
    has_dist_features = all(col in df.columns for col in ['dist1', 'dist2'])
    has_addr_features = all(col in df.columns for col in ['addr1', 'addr2'])
    
    if not (has_dist_features or has_addr_features):
        logger.warning("No geospatial columns found, skipping geospatial features.")
        return df
    
    # Process each customer separately
    unique_customers = df['card1'].unique()
    
    # Distance statistics (dist1, dist2 represent distances between billing address and seller location)
    if has_dist_features:
        # For each distance feature, compute customer-specific statistics
        for dist_col in ['dist1', 'dist2']:
            if dist_col in df.columns:
                # Compute statistics per customer
                dist_stats = df.groupby('card1')[dist_col].agg(['mean', 'std', 'min', 'max']).fillna(0)
                
                # Map back to original DataFrame
                df[f'cust_{dist_col}_mean'] = df['card1'].map(dist_stats['mean'])
                df[f'cust_{dist_col}_std'] = df['card1'].map(dist_stats['std'])
                
                # Compute z-score of distance for each customer
                eps = 1e-6  # Avoid division by zero
                df[f'{dist_col}_zscore'] = df.groupby('card1').apply(
                    lambda x: (x[dist_col] - x[dist_col].mean()) / (x[dist_col].std() + eps)
                ).values
                
                # Flag unusually large distances (more than 2 std devs from customer mean)
                df[f'{dist_col}_unusual'] = (df[f'{dist_col}_zscore'] > 2).astype('uint8')
    
    # Address clusters (addr1, addr2 represent anonymized seller location)
    if has_addr_features:
        # Initialize location cluster features
        df['location_cluster'] = -1
        df['is_common_location'] = 0
        df['location_frequency'] = 0
        
        # Process each customer
        for customer_id in unique_customers:
            # Filter data for this customer
            customer_mask = df['card1'] == customer_id
            customer_df = df[customer_mask].copy()
            
            # Skip if too few transactions
            if len(customer_df) < 5:
                continue
            
            try:
                # Extract location coordinates
                locations = customer_df[['addr1', 'addr2']].values
                
                # Handle missing values
                if np.isnan(locations).any():
                    locations = np.nan_to_num(locations, nan=-999)
                
                # Cluster locations using DBSCAN
                clustering = DBSCAN(eps=0.5, min_samples=2).fit(locations)
                clusters = clustering.labels_
                
                # Store cluster labels
                df.loc[customer_mask, 'location_cluster'] = clusters
                
                # Count transactions per cluster
                cluster_counts = pd.Series(clusters).value_counts()
                cluster_freq = cluster_counts / len(clusters)
                
                # Identify main clusters (most frequent locations)
                main_clusters = cluster_counts.nlargest(3).index.tolist()
                
                # Flag common locations
                df.loc[customer_mask, 'is_common_location'] = df.loc[customer_mask, 'location_cluster'].isin(main_clusters).astype('uint8')
                
                # Map cluster frequency as a feature
                cluster_freq_map = dict(zip(cluster_freq.index, cluster_freq.values))
                df.loc[customer_mask, 'location_frequency'] = df.loc[customer_mask, 'location_cluster'].map(
                    lambda x: cluster_freq_map.get(x, 0)
                )
            except Exception as e:
                logger.warning(f"Error clustering locations for customer {customer_id}: {str(e)}")
    
    # Geospatial velocity (if time and location data available)
    if 'TransactionDT' in df.columns and has_dist_features:
        # Calculate the rate of distance change over time
        for customer_id in unique_customers:
            # Filter and sort by time
            customer_mask = df['card1'] == customer_id
            customer_df = df[customer_mask].sort_values('TransactionDT')
            
            # Skip if too few transactions
            if len(customer_df) < 3:
                continue
            
            try:
                # Get transaction times in hours
                times = customer_df['TransactionDT'].values / 3600
                
                # Get distances
                for dist_col in ['dist1', 'dist2']:
                    if dist_col in df.columns:
                        distances = customer_df[dist_col].values
                        
                        # Calculate time and distance differences
                        time_diffs = np.diff(times)
                        dist_diffs = np.diff(distances)
                        
                        # Avoid division by zero
                        time_diffs = np.maximum(time_diffs, 0.001)
                        
                        # Calculate velocity (distance change per hour)
                        velocities = dist_diffs / time_diffs
                        
                        # Pad the first value (no previous transaction)
                        velocities = np.insert(velocities, 0, 0)
                        
                        # Store as feature
                        df.loc[customer_mask, f'{dist_col}_velocity'] = customer_df.index.map(
                            dict(zip(customer_df.index, velocities))
                        )
                        
                        # Flag unusually high velocities (top 10%)
                        high_velocity_threshold = np.percentile(velocities[velocities != 0], 90)
                        df.loc[customer_mask, f'high_{dist_col}_velocity'] = (
                            df.loc[customer_mask, f'{dist_col}_velocity'] > high_velocity_threshold
                        ).astype('uint8')
            except Exception as e:
                logger.warning(f"Error calculating geospatial velocity for customer {customer_id}: {str(e)}")
    
    logger.info("Customer geospatial features added.")
    return df

def add_customer_consistency_measures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features measuring the consistency of customer behavior.
    
    Args:
        df: DataFrame with card1 and transaction data
        
    Returns:
        DataFrame with added consistency measure features
    """
    logger.info("Adding customer consistency measures...")
    
    # Group by card1
    grouped = df.groupby('card1')
    
    # Entropy of transaction hour distribution (higher = more random)
    if 'hour' in df.columns:
        hour_entropy = grouped['hour'].apply(
            lambda x: stats.entropy(x.value_counts(normalize=True))
        )
        df['cust_hour_entropy'] = df['card1'].map(hour_entropy)
    
    # Entropy of transaction day distribution
    if 'dow' in df.columns:
        dow_entropy = grouped['dow'].apply(
            lambda x: stats.entropy(x.value_counts(normalize=True))
        )
        df['cust_dow_entropy'] = df['card1'].map(dow_entropy)
    
    # Consistency in transaction amounts
    if 'TransactionAmt' in df.columns:
        # Gini coefficient as a measure of amount inequality
        def gini(x):
            # Calculate Gini coefficient for array x
            x = np.sort(x)
            n = len(x)
            if n == 0:
                return 0
            cumx = np.cumsum(x, dtype=float)
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
        
        amount_gini = grouped['TransactionAmt'].apply(gini)
        df['cust_amt_gini'] = df['card1'].map(amount_gini)
        
        # Count unique transaction amounts
        unique_amounts = grouped['TransactionAmt'].apply(lambda x: x.nunique())
        df['cust_unique_amt_count'] = df['card1'].map(unique_amounts)
        
        # Ratio of unique amounts to total transactions
        df['cust_amt_uniqueness_ratio'] = df['cust_unique_amt_count'] / df['cust_txn_count']
    
    # Consistency in distances
    for dist_col in ['dist1', 'dist2']:
        if dist_col in df.columns:
            # Calculate entropy of distances
            dist_entropy = grouped[dist_col].apply(
                lambda x: stats.entropy(x.value_counts(bins=10, normalize=True))
            )
            df[f'cust_{dist_col}_entropy'] = df['card1'].map(dist_entropy)
    
    # Email and device consistency
    if all(col in df.columns for col in ['card1', 'P_emaildomain']):
        # Count unique email domains used by customer
        unique_emails = grouped['P_emaildomain'].apply(lambda x: x.nunique())
        df['cust_unique_email_count'] = df['card1'].map(unique_emails)
    
    if all(col in df.columns for col in ['card1', 'DeviceType']):
        # Count unique device types used by customer
        unique_devices = grouped['DeviceType'].apply(lambda x: x.nunique())
        df['cust_unique_device_count'] = df['card1'].map(unique_devices)
    
    logger.info("Customer consistency measures added.")
    return df

def add_customer_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features measuring deviation from customer's historical patterns.
    
    Args:
        df: DataFrame with card1 and transaction data
        
    Returns:
        DataFrame with added deviation features
    """
    logger.info("Adding customer pattern deviation features...")
    
    # Sort by customer and transaction time
    if all(col in df.columns for col in ['card1', 'TransactionDT']):
        df = df.sort_values(['card1', 'TransactionDT']).reset_index(drop=True)
        
        # Calculate features based on sequence within customer
        customers = df['card1'].unique()
        
        # Initialize deviation features
        df['txn_seq_number'] = 0  # Transaction sequence number
        df['time_since_prev_txn'] = 0.0  # Hours since previous transaction
        df['amt_diff_from_prev'] = 0.0  # Difference from previous amount
        df['amt_ratio_to_prev'] = 0.0  # Ratio to previous amount
        
        for customer in customers:
            cust_mask = df['card1'] == customer
            cust_df = df[cust_mask]
            
            if len(cust_df) <= 1:
                continue
            
            # Transaction sequence number
            df.loc[cust_mask, 'txn_seq_number'] = range(1, len(cust_df) + 1)
            
            # Time since previous transaction (in hours)
            prev_times = cust_df['TransactionDT'].shift(1)
            time_diffs = (cust_df['TransactionDT'] - prev_times) / 3600  # Convert to hours
            df.loc[cust_mask, 'time_since_prev_txn'] = time_diffs
            
            # Amount difference features (if TransactionAmt exists)
            if 'TransactionAmt' in df.columns:
                prev_amounts = cust_df['TransactionAmt'].shift(1)
                
                # Absolute difference
                amt_diffs = cust_df['TransactionAmt'] - prev_amounts
                df.loc[cust_mask, 'amt_diff_from_prev'] = amt_diffs
                
                # Ratio (with epsilon to avoid division by zero)
                eps = 1e-6
                amt_ratios = cust_df['TransactionAmt'] / (prev_amounts + eps)
                df.loc[cust_mask, 'amt_ratio_to_prev'] = amt_ratios
            
            # Distance difference features
            for dist_col in ['dist1', 'dist2']:
                if dist_col in df.columns:
                    prev_dist = cust_df[dist_col].shift(1)
                    dist_diffs = cust_df[dist_col] - prev_dist
                    df.loc[cust_mask, f'{dist_col}_diff_from_prev'] = dist_diffs
    
    # Flag anomalous time gaps
    if 'time_since_prev_txn' in df.columns:
        # Get customer-specific median time between transactions
        cust_median_time = df.groupby('card1')['time_since_prev_txn'].median()
        df['cust_median_time_gap'] = df['card1'].map(cust_median_time)
        
        # Calculate ratio of current gap to median gap
        eps = 1e-6  # To avoid division by zero
        df['time_gap_ratio'] = df['time_since_prev_txn'] / (df['cust_median_time_gap'] + eps)
        
        # Flag unusually long gaps (more than 5x median)
        df['is_unusual_time_gap'] = (df['time_gap_ratio'] > 5).astype('uint8')
    
    # Flag anomalous amounts relative to history
    if 'TransactionAmt' in df.columns and 'cust_amt_mean' in df.columns:
        # Flag unusually large amounts (more than 3 std devs from mean)
        eps = 1e-6  # To avoid division by zero
        df['amt_std_from_mean'] = (df['TransactionAmt'] - df['cust_amt_mean']) / (df['cust_amt_std'] + eps)
        df['is_unusual_amount'] = (abs(df['amt_std_from_mean']) > 3).astype('uint8')
    
    logger.info("Customer pattern deviation features added.")
    return df

def build_customer_features(input_path: str, output_path: str) -> None:
    """
    Load data, add customer-specific features, and save to output file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output CSV with added features
    """
    logger.info(f"Building customer features: {input_path} -> {output_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Add features
    df = add_customer_features(df)
    
    # Save result
    df.to_csv(output_path, index=False)
    
    logger.info(f"Customer features built and saved to {output_path}")
    
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Build customer features for fraud detection")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    
    args = parser.parse_args()
    
    # Run feature builder
    build_customer_features(args.input, args.output)