"""
Preprocessing utilities: missing-value imputation, label encoding,
and downcasting numeric types to save memory.
"""

import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast floats to float32 and ints to the smallest unsigned subtype.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with smaller numeric dtypes.
    """
    for c in df.select_dtypes(include=["float64"]):
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]):
        df[c] = pd.to_numeric(df[c], downcast="unsigned")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values, encode categorical email domains, drop unused cols,
    then downcast numeric types.

    Args:
        df: Raw merged DataFrame with both transaction & identity columns.

    Returns:
        Cleaned DataFrame ready for feature engineering / modeling.
    """
    df = df.copy()
    logger.info("Starting preprocessing.")

    # Identity columns
    id_cols = [c for c in df.columns if c.startswith("id_")]
    for c in id_cols:
        df[c].fillna(df[c].median() if df[c].dtype != object else "unknown", inplace=True)

    # Email domains
    for c in ("P_emaildomain", "R_emaildomain"):
        if c in df:
            df[c].fillna("unknown", inplace=True)
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))

    # Numeric columns
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)

    # Drop IDs and raw DT
    df.drop(["TransactionID", "TransactionDT"], axis=1, errors="ignore", inplace=True)

    # Downcast
    df = downcast_df(df)
    logger.info(f"Preprocessed DataFrame shape: {df.shape}")
    return df