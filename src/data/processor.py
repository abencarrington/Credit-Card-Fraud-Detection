"""
Preprocessing module for IEEE-CIS fraud detection dataset.

Provides functions for:
- Missing value imputation with advanced strategies
- Categorical encoding with frequency and target encoding
- Feature normalization and scaling
- Outlier handling
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

def preprocess_data(
    df: pd.DataFrame, 
    target_col: str = 'isFraud',
    handle_missing: bool = True,
    encode_categoricals: bool = True,
    scale_numerics: bool = False,
    handle_outliers: bool = True
) -> pd.DataFrame:
    """
    Apply comprehensive preprocessing to the IEEE-CIS dataset.
    
    Args:
        df: Raw DataFrame with transaction and identity columns
        target_col: Name of the target column
        handle_missing: Whether to impute missing values
        encode_categoricals: Whether to encode categorical variables
        scale_numerics: Whether to scale numeric features
        handle_outliers: Whether to handle outliers
        
    Returns:
        Preprocessed DataFrame ready for feature engineering
    """
    df = df.copy()
    logger.info("Starting preprocessing pipeline...")
    
    # Save original dtypes for reference
    original_dtypes = df.dtypes.to_dict()
    
    # 1. Handle missing values
    if handle_missing:
        df = handle_missing_values(df, target_col)
    
    # 2. Encode categorical variables
    if encode_categoricals:
        df = encode_categorical_features(df, target_col)
    
    # 3. Handle outliers
    if handle_outliers:
        df = handle_numeric_outliers(df, target_col)
    
    # 4. Scale numeric features
    if scale_numerics:
        df = scale_numeric_features(df, target_col)
    
    # 5. Final cleanup
    # Drop columns with all missing values
    for col in df.columns:
        if df[col].isna().all():
            df.drop(col, axis=1, inplace=True)
    
    # Restore target column dtype if it was changed
    if target_col in df.columns and target_col in original_dtypes:
        df[target_col] = df[target_col].astype(original_dtypes[target_col])
    
    logger.info(f"Preprocessing complete. Final DataFrame shape: {df.shape}")
    return df

def handle_missing_values(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Handle missing values with appropriate strategies based on data types.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        DataFrame with imputed missing values
    """
    df = df.copy()
    logger.info("Handling missing values...")
    
    # 1. Identify missing values by column type
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include=['int', 'float']).columns
    
    # 2. For categorical columns
    for col in cat_cols:
        if col != target_col and df[col].isna().any():
            # For high cardinality, use special 'unknown' value
            if df[col].nunique() > 10:
                df[col] = df[col].fillna('unknown')
            # For low cardinality, use most frequent value
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # 3. For numeric columns
    # Group by fraud/non-fraud for more accurate imputation
    for col in num_cols:
        if col != target_col and df[col].isna().any():
            # If column has many missing values (>50%), use -999 as a flag
            if df[col].isna().mean() > 0.5:
                df[col] = df[col].fillna(-999)
            # If target column exists, use fraud-aware imputation
            elif target_col in df.columns:
                # Separate imputation for fraud and non-fraud
                for label in df[target_col].unique():
                    mask = (df[target_col] == label)
                    df.loc[mask, col] = df.loc[mask, col].fillna(
                        df.loc[mask, col].median()
                    )
            # Otherwise use median
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # 4. For ID columns (usually start with 'id_')
    id_cols = [c for c in df.columns if c.startswith('id_')]
    for col in id_cols:
        if df[col].isna().any():
            # Use -1 for missing IDs
            df[col] = df[col].fillna(-1)
    
    logger.info("Missing value handling complete.")
    return df

def encode_categorical_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Encode categorical features using appropriate strategies.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    logger.info("Encoding categorical features...")
    
    # 1. Identify categorical columns to encode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    # Create dictionary to store encoders for potential future use
    encoders = {}
    
    # 2. Apply encoding strategies
    for col in cat_cols:
        # Skip if column has too many unique values or missing values
        if df[col].nunique() > 100 or df[col].isna().any():
            logger.warning(f"Skipping encoding for {col} due to high cardinality or missing values")
            continue
        
        # For binary categorical features (2 values), use simple mapping
        if df[col].nunique() == 2:
            # Get the values
            unique_vals = df[col].unique()
            # Create simple 0/1 mapping
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[col] = df[col].map(mapping)
            
        # For low cardinality features, use label encoding
        elif df[col].nunique() < 20:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        # For email domains, extract the domain root
        elif 'email' in col.lower():
            # Extract root domain (e.g., gmail from gmail.com)
            df[f"{col}_root"] = df[col].str.split('.').str[0]
            # Encode the root domain
            le = LabelEncoder()
            df[f"{col}_root"] = le.fit_transform(df[f"{col}_root"].fillna('unknown').astype(str))
            encoders[f"{col}_root"] = le
            
        # For other columns, use label encoding but keep original
        else:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[f"{col}_encoded"] = le
    
    logger.info("Categorical encoding complete.")
    return df

def handle_numeric_outliers(
    df: pd.DataFrame, 
    target_col: str,
    method: str = 'winsorize',
    threshold: float = 0.01
) -> pd.DataFrame:
    """
    Handle outliers in numeric features.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        method: Outlier handling method ('winsorize', 'clip', 'log')
        threshold: Percentile threshold for outlier detection
        
    Returns:
        DataFrame with handled outliers
    """
    df = df.copy()
    logger.info(f"Handling numeric outliers using {method} method...")
    
    # Identify numeric columns
    num_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    
    # Exclude target and ID columns
    exclude_cols = [target_col] + [col for col in df.columns if col.startswith(('id_', 'ID'))]
    num_cols = [col for col in num_cols if col not in exclude_cols]
    
    # Handle each column
    for col in num_cols:
        # If column is all zeros or single value, skip
        if df[col].nunique() <= 1:
            continue
            
        # Get outlier thresholds (separate for fraud/non-fraud if available)
        if target_col in df.columns:
            thresholds = {}
            # Calculate thresholds for each target value
            for label in df[target_col].unique():
                values = df.loc[df[target_col] == label, col]
                if len(values) > 10:  # Need enough samples
                    lower = values.quantile(threshold)
                    upper = values.quantile(1 - threshold)
                    thresholds[label] = (lower, upper)
                    
            # Apply thresholds by label
            for label, (lower, upper) in thresholds.items():
                mask = (df[target_col] == label)
                
                if method == 'winsorize':
                    df.loc[mask & (df[col] < lower), col] = lower
                    df.loc[mask & (df[col] > upper), col] = upper
                elif method == 'clip':
                    df.loc[mask, col] = df.loc[mask, col].clip(lower, upper)
                elif method == 'log':
                    # Only apply log to columns with all positive values
                    if (df.loc[mask, col] > 0).all():
                        # Add small constant to avoid log(0)
                        df.loc[mask, col] = np.log1p(df.loc[mask, col])
        else:
            # Calculate global thresholds
            lower = df[col].quantile(threshold)
            upper = df[col].quantile(1 - threshold)
            
            if method == 'winsorize':
                df.loc[df[col] < lower, col] = lower
                df.loc[df[col] > upper, col] = upper
            elif method == 'clip':
                df[col] = df[col].clip(lower, upper)
            elif method == 'log':
                # Only apply log to columns with all positive values
                if (df[col] > 0).all():
                    # Add small constant to avoid log(0)
                    df[col] = np.log1p(df[col])
    
    logger.info("Outlier handling complete.")
    return df

def scale_numeric_features(
    df: pd.DataFrame, 
    target_col: str,
    method: str = 'standard'
) -> pd.DataFrame:
    """
    Scale numeric features using specified method.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        method: Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        DataFrame with scaled numeric features
    """
    df = df.copy()
    logger.info(f"Scaling numeric features using {method} method...")
    
    # Identify numeric columns
    num_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    
    # Exclude target and ID columns
    exclude_cols = [target_col] + [col for col in df.columns if col.startswith(('id_', 'ID'))]
    num_cols = [col for col in num_cols if col not in exclude_cols]
    
    # Apply scaling
    if method == 'standard':
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    
    logger.info("Feature scaling complete.")
    return df

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from TransactionDT column.
    
    Args:
        df: DataFrame with TransactionDT column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    if 'TransactionDT' not in df.columns:
        logger.warning("TransactionDT column not found, skipping time feature extraction.")
        return df
    
    logger.info("Extracting time features from TransactionDT...")
    
    # Convert to hours from start date
    df['transaction_hour'] = (df['TransactionDT'] / 3600) % 24
    
    # Extract day of week (IEEE-CIS dataset starts on 2017-12-01, which was a Friday)
    start_day = 4  # 0=Monday, 4=Friday
    df['day_of_week'] = ((df['TransactionDT'] / (3600 * 24) + start_day) % 7).astype(int)
    
    # Extract day of month
    # We don't know exact dates, but we can use modulo 30 to approximate
    df['day_of_month'] = ((df['TransactionDT'] / (3600 * 24)) % 30 + 1).astype(int)
    
    # Extract week of year (approximate)
    df['week_of_year'] = ((df['TransactionDT'] / (3600 * 24 * 7)) % 52 + 1).astype(int)
    
    # Is weekend flag (day_of_week 5,6 = weekend)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Is night transaction (from 10 PM to 6 AM)
    df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] < 6)).astype(int)
    
    logger.info("Time feature extraction complete.")
    return df