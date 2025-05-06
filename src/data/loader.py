"""
Data loader module for the IEEE-CIS fraud detection dataset.

Provides functions to efficiently read transaction and identity CSVs from the
IEEE-CIS Fraud Detection dataset and merge them with optimized memory usage.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric columns to appropriate types.
    
    Args:
        df: Input DataFrame to optimize
        
    Returns:
        DataFrame with optimized dtypes
    """
    result = df.copy()
    
    # Downcast integer columns
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        # Determine min/max to find appropriate int type
        col_min, col_max = df[col].min(), df[col].max()
        
        # Check if unsigned
        if col_min >= 0:
            if col_max < 2**8:
                result[col] = df[col].astype(np.uint8)
            elif col_max < 2**16:
                result[col] = df[col].astype(np.uint16)
            elif col_max < 2**32:
                result[col] = df[col].astype(np.uint32)
            else:
                result[col] = df[col].astype(np.uint64)
        else:
            if col_min > -2**7 and col_max < 2**7:
                result[col] = df[col].astype(np.int8)
            elif col_min > -2**15 and col_max < 2**15:
                result[col] = df[col].astype(np.int16)
            elif col_min > -2**31 and col_max < 2**31:
                result[col] = df[col].astype(np.int32)
            else:
                result[col] = df[col].astype(np.int64)
    
    # Downcast float columns
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        result[col] = df[col].astype(np.float32)
    
    return result

def downcast_csv(input_path: str, output_path: str) -> None:
    """
    Read a CSV, downcast its numeric columns, and save to a new file.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output downcasted CSV
    """
    logger.info(f"Downcasting {input_path} -> {output_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read in chunks to handle large files
    chunk_size = 100000
    chunks = pd.read_csv(input_path, chunksize=chunk_size)
    
    # Process first chunk to get column names and dtypes
    first_chunk = next(chunks)
    first_chunk = downcast_dtypes(first_chunk)
    
    # Write first chunk
    first_chunk.to_csv(output_path, index=False, mode='w')
    
    # Process remaining chunks
    for chunk in chunks:
        chunk = downcast_dtypes(chunk)
        chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    logger.info(f"Downcasting complete. Output saved to {output_path}")

def load_transaction_data(data_dir: Union[str, Path], train: bool = True) -> pd.DataFrame:
    """
    Load the IEEE-CIS transaction CSV with optimized memory usage.
    
    Args:
        data_dir: Path to the folder containing train/test CSVs
        train: If True, loads train_transaction.csv; else test_transaction.csv
        
    Returns:
        DataFrame of transaction data
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
        
    fname = "train_transaction.csv" if train else "test_transaction.csv"
    path = data_dir / fname
    
    # Check if downcasted version exists
    processed_dir = data_dir.parent / "processed"
    downcasted_path = processed_dir / f"{fname.replace('.csv', '_downcast.csv')}"
    
    if downcasted_path.exists():
        logger.info(f"Loading downcasted transactions from {downcasted_path}")
        return pd.read_csv(downcasted_path)
    
    logger.info(f"Loading transactions from {path}")
    
    # Define dtypes to reduce memory usage
    dtypes = {
        'TransactionID': 'uint32',
        'isFraud': 'uint8',
        'TransactionDT': 'uint32',
        'TransactionAmt': 'float32',
        'ProductCD': 'category',
        'card1': 'int32',
        'card2': 'float32',
        'card3': 'float32',
        'card4': 'category',
        'card5': 'float32',
        'card6': 'category',
        'addr1': 'float32',
        'addr2': 'float32',
        'dist1': 'float32',
        'dist2': 'float32',
        'P_emaildomain': 'category',
        'R_emaildomain': 'category',
    }
    
    # Read data
    try:
        df = pd.read_csv(path, dtype=dtypes)
        
        # Downcast remaining columns
        df = downcast_dtypes(df)
        
        # Create processed directory if needed
        processed_dir.mkdir(exist_ok=True, parents=True)
        
        # Save downcasted version for future use
        logger.info(f"Saving downcasted version to {downcasted_path}")
        df.to_csv(downcasted_path, index=False)
        
        return df
    except Exception as e:
        logger.error(f"Error loading transaction data: {str(e)}")
        raise

def load_identity_data(data_dir: Union[str, Path], train: bool = True) -> pd.DataFrame:
    """
    Load the IEEE-CIS identity CSV with optimized memory usage.
    
    Args:
        data_dir: Path to the folder containing train/test CSVs
        train: If True, loads train_identity.csv; else test_identity.csv
        
    Returns:
        DataFrame of identity data
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
        
    fname = "train_identity.csv" if train else "test_identity.csv"
    path = data_dir / fname
    
    # Check if downcasted version exists
    processed_dir = data_dir.parent / "processed"
    downcasted_path = processed_dir / f"{fname.replace('.csv', '_downcast.csv')}"
    
    if downcasted_path.exists():
        logger.info(f"Loading downcasted identity data from {downcasted_path}")
        return pd.read_csv(downcasted_path)
    
    logger.info(f"Loading identity data from {path}")
    
    # Define dtypes to reduce memory usage
    dtypes = {
        'TransactionID': 'uint32',
        'DeviceType': 'category',
        'DeviceInfo': 'category',
    }
    
    # Identify ID columns for categorical type
    id_cols = [f'id_{i}' for i in range(1, 39)]
    for col in id_cols:
        dtypes[col] = 'category'
    
    # Read data
    try:
        df = pd.read_csv(path, dtype=dtypes)
        
        # Downcast remaining columns
        df = downcast_dtypes(df)
        
        # Create processed directory if needed
        processed_dir.mkdir(exist_ok=True, parents=True)
        
        # Save downcasted version for future use
        logger.info(f"Saving downcasted version to {downcasted_path}")
        df.to_csv(downcasted_path, index=False)
        
        return df
    except Exception as e:
        logger.error(f"Error loading identity data: {str(e)}")
        raise

def load_data(data_dir: Union[str, Path], train: bool = True) -> pd.DataFrame:
    """
    Load and merge transaction and identity tables from IEEE-CIS dataset.
    
    Args:
        data_dir: Path to the folder containing CSVs
        train: Load train vs. test data
        
    Returns:
        DataFrame with merged transaction and identity data
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    # Check if merged file already exists
    processed_dir = data_dir.parent / "processed"
    merged_filename = f"{'train' if train else 'test'}_merged_downcast.csv"
    merged_path = processed_dir / merged_filename
    
    if merged_path.exists():
        logger.info(f"Loading pre-merged data from {merged_path}")
        return pd.read_csv(merged_path)
    
    # Load separate tables
    trans = load_transaction_data(data_dir, train)
    
    try:
        ident = load_identity_data(data_dir, train)
        logger.info(f"Merging {len(trans)} transactions with {len(ident)} identities")
        df = trans.merge(ident, on="TransactionID", how="left")
    except FileNotFoundError:
        logger.warning(f"Identity file not found. Using transaction data only.")
        df = trans
    
    # Save merged file for future use
    processed_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving merged data to {merged_path}")
    df.to_csv(merged_path, index=False)
    
    logger.info(f"Resulting DataFrame shape: {df.shape}")
    return df

def get_dataset_stats(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Get statistics about the IEEE-CIS dataset.
    
    Args:
        data_dir: Path to the folder containing CSVs
        
    Returns:
        Dictionary of dataset statistics
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    stats = {}
    
    try:
        # Load a sample to get column info
        trans_sample = pd.read_csv(
            data_dir / "train_transaction.csv", 
            nrows=1000
        )
        
        stats["transaction_columns"] = trans_sample.columns.tolist()
        stats["transaction_column_count"] = len(trans_sample.columns)
        
        # Count rows without loading full dataset
        with open(data_dir / "train_transaction.csv", 'r') as f:
            # Subtract 1 for header
            stats["transaction_row_count"] = sum(1 for _ in f) - 1
        
        # Try to get identity stats
        try:
            ident_sample = pd.read_csv(
                data_dir / "train_identity.csv", 
                nrows=1000
            )
            
            stats["identity_columns"] = ident_sample.columns.tolist()
            stats["identity_column_count"] = len(ident_sample.columns)
            
            with open(data_dir / "train_identity.csv", 'r') as f:
                stats["identity_row_count"] = sum(1 for _ in f) - 1
        except FileNotFoundError:
            stats["identity_available"] = False
        else:
            stats["identity_available"] = True
        
        # Check for test data
        stats["test_available"] = (data_dir / "test_transaction.csv").exists()
        
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}")
        return {"error": str(e)}
    
    return stats