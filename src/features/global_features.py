"""
Global features module for IEEE-CIS fraud detection.

Implements global-level features including:
- Time-based features
- Transaction amount features
- Card usage patterns
- Distance features (but excluding addr1/addr2 as including these for nationwide customers will muddy results)
- Transaction velocity at different time windows
- Email and device fingerprinting
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from scipy import stats

logger = logging.getLogger(__name__)

def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add global features applicable to all transactions.
    
    Includes:
    - Time features (hour, day, weekday)
    - Transaction amount features (log, bins, z-score)
    - Card usage patterns
    - Distance features (dist1, dist2, but not addr1/addr2)
    - Transaction velocity at different time windows
    - Email and device type features
    
    Args:
        df: DataFrame with raw transaction data
        
    Returns:
        DataFrame with added global features
    """
    df = df.copy()
    logger.info("Adding global features...")
    
    # Apply all feature transformations
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_card_features(df)
    df = add_distance_features(df)  # Add distance but not address features
    df = add_velocity_features(df)
    df = add_email_device_features(df)
    
    logger.info(f"Added global features. New DataFrame shape: {df.shape}")
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features derived from TransactionDT.
    
    Args:
        df: DataFrame with TransactionDT column
        
    Returns:
        DataFrame with added time features
    """
    if 'TransactionDT' not in df.columns:
        logger.warning("TransactionDT column not found, skipping time features.")
        return df
    
    logger.info("Adding time-based features...")
    
    # IEEE-CIS dataset's first day is 2017-12-01 (a Friday)
    # Convert seconds to days (floating point)
    days_from_start = df['TransactionDT'] / (3600 * 24)
    
    # Hour of day (0-23)
    df['hour'] = ((df['TransactionDT'] / 3600) % 24).astype('uint8')
    
    # Day of week (0=Monday, 6=Sunday)
    # Start date was Friday (4), so we add 4 and take modulo 7
    df['dow'] = ((days_from_start + 4) % 7).astype('uint8')
    
    # Day of month (1-31)
    # This is approximate since we don't know exact calendar date
    df['day'] = ((days_from_start % 30) + 1).astype('uint8')
    
    # Month (0-11, for 6 months of data)
    df['month'] = (days_from_start / 30).astype('uint8')
    
    # Hour categories: morning, afternoon, evening, night
    # 5-11 = morning, 12-16 = afternoon, 17-21 = evening, 22-4 = night
    hour_cats = pd.cut(
        df['hour'], 
        bins=[-1, 4, 11, 16, 21, 24], 
        labels=['night', 'morning', 'afternoon', 'evening', 'night']
    )
    df['hour_category'] = hour_cats
    
    # Is weekend
    df['is_weekend'] = df['dow'].isin([5, 6]).astype('uint8')
    
    # Time period features
    # Define business hours (9-17 on weekdays)
    business_hours = (df['hour'].between(9, 17)) & (~df['is_weekend'])
    df['is_business_hours'] = business_hours.astype('uint8')
    
    # Define night hours (22-6)
    night_hours = (df['hour'] >= 22) | (df['hour'] <= 6)
    df['is_night'] = night_hours.astype('uint8')
    
    logger.info("Time features added.")
    return df

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transaction amount related features.
    
    Args:
        df: DataFrame with TransactionAmt column
        
    Returns:
        DataFrame with added amount features
    """
    if 'TransactionAmt' not in df.columns:
        logger.warning("TransactionAmt column not found, skipping amount features.")
        return df
    
    logger.info("Adding transaction amount features...")
    
    # Log transform to handle skewed distribution
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt']).astype('float32')
    
    # Amount bins (10 bins based on quantiles)
    df['TransactionAmt_bin'] = pd.qcut(
        df['TransactionAmt'], 
        q=10, 
        labels=False, 
        duplicates='drop'
    ).astype('uint8')
    
    # Rounded amount features
    df['TransactionAmt_round_10'] = np.round(df['TransactionAmt'] / 10) * 10
    df['TransactionAmt_round_100'] = np.round(df['TransactionAmt'] / 100) * 100
    
    # Is round amount flag
    df['is_round_amount'] = ((df['TransactionAmt'] == df['TransactionAmt_round_10']) | 
                            (df['TransactionAmt'] == df['TransactionAmt_round_100'])).astype('uint8')
    
    # Cents part of amount
    df['cents'] = np.round((df['TransactionAmt'] - np.floor(df['TransactionAmt'])) * 100)
    df['has_cents'] = (df['cents'] > 0).astype('uint8')
    
    # Amount fractional digits
    df['amount_decimal_len'] = df['TransactionAmt'].astype(str).str.split('.').str[1].str.len()
    df['amount_decimal_len'] = df['amount_decimal_len'].fillna(0).astype('uint8')
    
    # Z-score of amount (globally)
    df['TransactionAmt_zscore'] = stats.zscore(df['TransactionAmt']).astype('float32')
    
    logger.info("Amount features added.")
    return df

def add_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add card-related features and aggregate statistics.
    
    Args:
        df: DataFrame with card1, card2, etc. columns
        
    Returns:
        DataFrame with added card features
    """
    logger.info("Adding card-related features...")
    
    # Check if card columns exist
    card_cols = [col for col in df.columns if col.startswith('card')]
    if not card_cols:
        logger.warning("No card columns found, skipping card features.")
        return df
    
    # Process each card feature if it exists
    # Card1 features
    if 'card1' in df.columns:
        # Number of transactions per card1
        card1_counts = df['card1'].value_counts()
        df['card1_count'] = df['card1'].map(card1_counts).astype('uint16')
        
        # Card1 amount statistics
        card1_amount_mean = df.groupby('card1')['TransactionAmt'].mean()
        card1_amount_std = df.groupby('card1')['TransactionAmt'].std().fillna(0)
        
        df['card1_amount_mean'] = df['card1'].map(card1_amount_mean).astype('float32')
        df['card1_amount_std'] = df['card1'].map(card1_amount_std).astype('float32')
        
        # Transaction amount relative to card1 mean
        eps = 1e-6  # To avoid division by zero
        df['amount_to_card1_mean'] = (df['TransactionAmt'] / (df['card1_amount_mean'] + eps)).astype('float32')
    
    # Card2 features (if exists)
    if 'card2' in df.columns:
        # Number of transactions per card2
        card2_counts = df['card2'].value_counts()
        df['card2_count'] = df['card2'].map(card2_counts).astype('uint16')
    
    # Card3 features (if exists)
    if 'card3' in df.columns:
        # Number of transactions per card3
        card3_counts = df['card3'].value_counts()
        df['card3_count'] = df['card3'].map(card3_counts).astype('uint16')
    
    # Card type features
    if 'card4' in df.columns:
        # Create dummies for card type
        card4_dummies = pd.get_dummies(df['card4'], prefix='card4')
        df = pd.concat([df, card4_dummies], axis=1)
    
    # Card category features
    if 'card6' in df.columns:
        # Create dummies for card category
        card6_dummies = pd.get_dummies(df['card6'], prefix='card6')
        df = pd.concat([df, card6_dummies], axis=1)
    
    # Card pairs: count unique combinations
    if all(col in df.columns for col in ['card1', 'card2']):
        # Group by combinations
        card_pairs = df.groupby(['card1', 'card2']).size().reset_index()
        card_pairs.columns = ['card1', 'card2', 'card_pair_count']
        
        # Map back to original DataFrame
        pair_dict = dict(zip(zip(card_pairs['card1'], card_pairs['card2']), card_pairs['card_pair_count']))
        df['card_pair_count'] = df.apply(lambda x: pair_dict.get((x['card1'], x['card2']), 0), axis=1)
    
    logger.info("Card features added.")
    return df

def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features based on distance variables (dist1, dist2) only.
    
    Note: We intentionally exclude addr1/addr2 features at the global level
    as they are more meaningful at the customer level. This function focuses
    only on the distance metrics which have global relevance.
    
    Args:
        df: DataFrame with dist1, dist2 columns
        
    Returns:
        DataFrame with added distance features
    """
    # Check if distance columns exist
    dist_cols = [col for col in df.columns if col.startswith('dist')]
    if not dist_cols:
        logger.warning("No distance columns found, skipping distance features.")
        return df
    
    logger.info("Adding distance-based features...")
    
    # Process each distance feature
    for col in dist_cols:
        # Skip columns with too many missing values
        if df[col].isna().mean() > 0.5:
            logger.warning(f"Skipping {col} due to >50% missing values")
            continue
        
        # Fill missing values with median
        df[col].fillna(df[col].median(), inplace=True)
        
        # Log transform to handle skewed distribution
        df[f'{col}_log'] = np.log1p(df[col].clip(lower=0)).astype('float32')
        
        # Bin distances into percentiles
        df[f'{col}_bin'] = pd.qcut(
            df[col], 
            q=10, 
            labels=False, 
            duplicates='drop'
        ).astype('uint8')
        
        # Flag zero distances (same location)
        df[f'is_zero_{col}'] = (df[col] == 0).astype('uint8')
        
        # Flag extreme distances (top 5%)
        df[f'is_extreme_{col}'] = (df[col] > df[col].quantile(0.95)).astype('uint8')
    
    # Combine dist1 and dist2 if both present
    if all(col in df.columns for col in ['dist1', 'dist2']):
        # Calculate Euclidean distance
        df['dist_euclidean'] = np.sqrt(df['dist1']**2 + df['dist2']**2).astype('float32')
        
        # Calculate ratio of dist1 to dist2
        eps = 1e-6  # To avoid division by zero
        df['dist_ratio'] = (df['dist1'] / (df['dist2'] + eps)).astype('float32')
    
    logger.info("Distance features added.")
    return df

def add_velocity_features(df: pd.DataFrame, windows: List[int] = [1, 6, 24, 72, 168]) -> pd.DataFrame:
    """
    Add transaction velocity features at different time windows.
    
    Args:
        df: DataFrame with TransactionDT and card1
        windows: List of time windows in hours
        
    Returns:
        DataFrame with added velocity features
    """
    if 'TransactionDT' not in df.columns or 'card1' not in df.columns:
        logger.warning("Required columns for velocity features not found.")
        return df
    
    logger.info("Adding transaction velocity features...")
    
    # Sort by card1 and transaction time
    df = df.sort_values(['card1', 'TransactionDT']).reset_index(drop=True)
    
    # Convert TransactionDT to hours
    df['TransactionDT_hours'] = df['TransactionDT'] / 3600
    
    # For each window, compute transaction count and amount sum
    for window in windows:
        # Initialize features
        count_col = f'velocity_{window}h'
        amount_col = f'amount_velocity_{window}h'
        
        df[count_col] = 0
        df[amount_col] = 0.0
        
        # Group by card1 and compute rolling counts/sums
        for card in df['card1'].unique():
            # Get indices for this card
            card_indices = df.index[df['card1'] == card].tolist()
            if len(card_indices) <= 1:
                continue
                
            # For each transaction, compute velocities
            for i, idx in enumerate(card_indices):
                current_time = df.loc[idx, 'TransactionDT_hours']
                window_start = current_time - window
                
                # Get previous transactions within window
                prev_indices = [
                    prev_idx for prev_idx in card_indices[:i] 
                    if df.loc[prev_idx, 'TransactionDT_hours'] >= window_start
                ]
                
                # Count transactions in window
                df.loc[idx, count_col] = len(prev_indices)
                
                # Sum amounts in window
                if len(prev_indices) > 0:
                    df.loc[idx, amount_col] = df.loc[prev_indices, 'TransactionAmt'].sum()
    
    # Drop temporary column
    df.drop('TransactionDT_hours', axis=1, inplace=True)
    
    logger.info("Velocity features added.")
    return df

def add_email_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add email domain and device fingerprinting features.
    
    Args:
        df: DataFrame with email and device columns
        
    Returns:
        DataFrame with added email and device features
    """
    logger.info("Adding email and device features...")
    
    # Email domain features
    email_cols = [col for col in df.columns if 'email' in col.lower()]
    for col in email_cols:
        if col in df.columns:
            # Extract domain root (e.g., 'gmail' from 'gmail.com')
            df[f'{col}_root'] = df[col].str.split('.').str[0].fillna('unknown')
            
            # Count transactions per email domain
            email_counts = df[col].value_counts()
            df[f'{col}_count'] = df[col].map(email_counts).astype('uint16')
            
            # Categorize domains
            popular_domains = ['gmail', 'yahoo', 'hotmail', 'aol', 'outlook']
            
            def categorize_domain(domain):
                if pd.isna(domain):
                    return 'unknown'
                root = str(domain).split('.')[0].lower()
                if root in popular_domains:
                    return 'popular'
                elif any(edu in str(domain).lower() for edu in ['edu', 'university', 'school']):
                    return 'education'
                elif any(corp in str(domain).lower() for corp in ['corp', 'company', 'ltd']):
                    return 'corporate'
                else:
                    return 'other'
            
            df[f'{col}_category'] = df[col].apply(categorize_domain)
    
    # Device features
    if 'DeviceType' in df.columns:
        # Create dummies for device type
        device_dummies = pd.get_dummies(df['DeviceType'], prefix='device')
        df = pd.concat([df, device_dummies], axis=1)
        
        # Count transactions per device type
        device_counts = df['DeviceType'].value_counts()
        df['device_count'] = df['DeviceType'].map(device_counts).astype('uint16')
    
    # Device info features
    if 'DeviceInfo' in df.columns:
        # Extract device brand (if available)
        df['device_brand'] = df['DeviceInfo'].str.split().str[0].fillna('unknown')
        
        # Check if device info contains version number
        df['has_version'] = df['DeviceInfo'].str.contains('[0-9]').fillna(False).astype('uint8')
    
    # Email-device consistency (if both present)
    if all(col in df.columns for col in ['P_emaildomain', 'DeviceType']):
        # Create combination key
        df['email_device_combo'] = df['P_emaildomain'].astype(str) + '_' + df['DeviceType'].astype(str)
        
        # Count combinations
        combo_counts = df['email_device_combo'].value_counts()
        df['email_device_combo_count'] = df['email_device_combo'].map(combo_counts).astype('uint16')
        
        # Flag rare combinations (potential anomalies)
        df['rare_email_device'] = (df['email_device_combo_count'] < 10).astype('uint8')
        
        # Clean up temporary column
        df.drop('email_device_combo', axis=1, inplace=True)
    
    logger.info("Email and device features added.")
    return df

def build_global_features(input_path: str, output_path: str) -> None:
    """
    Load data, add global features, and save to output file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output CSV with added features
    """
    logger.info(f"Building global features: {input_path} -> {output_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Add features
    df = add_global_features(df)
    
    # Save result
    df.to_csv(output_path, index=False)
    
    logger.info(f"Global features built and saved to {output_path}")
    
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Build global features for fraud detection")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    
    args = parser.parse_args()
    
    # Run feature builder
    build_global_features(args.input, args.output)