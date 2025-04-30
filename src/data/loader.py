"""
Data loader module for the IEEE-CIS fraud detection dataset.

Provides functions to read transaction and identity CSVs and merge them.
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_transaction_data(data_dir: str, train: bool = True) -> pd.DataFrame:
    """
    Load the transaction CSV.

    Args:
        data_dir: Path to the folder containing train/test CSVs.
        train:   If True, loads train_transaction.csv; else test_transaction.csv.

    Returns:
        A DataFrame of transaction data.
    """
    fname = "train_transaction.csv" if train else "test_transaction.csv"
    path = os.path.join(data_dir, fname)
    logger.info(f"Loading transactions from {path}")
    return pd.read_csv(path)

def load_identity_data(data_dir: str, train: bool = True) -> pd.DataFrame:
    """
    Load the identity CSV.

    Args:
        data_dir: Path to the folder containing train/test CSVs.
        train:   If True, loads train_identity.csv; else test_identity.csv.

    Returns:
        A DataFrame of identity data.
    """
    fname = "train_identity.csv" if train else "test_identity.csv"
    path = os.path.join(data_dir, fname)
    logger.info(f"Loading identities from {path}")
    return pd.read_csv(path)

def load_data(data_dir: str, train: bool = True) -> pd.DataFrame:
    """
    Merge transaction and identity tables on TransactionID.

    Args:
        data_dir: Path to the folder containing CSVs.
        train:   Load train vs. test.

    Returns:
        A single merged DataFrame.
    """
    trans = load_transaction_data(data_dir, train)
    ident = load_identity_data(data_dir, train)
    logger.info(f"Merging {len(trans)} transactions with {len(ident)} identities")
    df = trans.merge(ident, on="TransactionID", how="left")
    logger.info(f"Resulting DataFrame shape: {df.shape}")
    return df