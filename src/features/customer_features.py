"""
Create per-customer features: counts, sums, std, and deviations,
so each VAE sees that customer’s own pattern.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-card1:
      - total transaction count
      - sum & std of TransactionAmt
      - deviation from card1 mean

    Args:
        df: DataFrame including a 'card1' and 'TransactionAmt'.

    Returns:
        DataFrame enriched with customer-specific features.
    """
    df = df.copy()
    logger.info("Adding customer-specific features.")

    grp = df.groupby("card1")["TransactionAmt"]
    df["cust_txn_count"]     = grp.transform("count").astype("uint16")
    df["cust_amt_sum"]       = grp.transform("sum").astype("float32")
    df["cust_amt_std"]       = grp.transform("std").fillna(0).astype("float32")
    df["amt_dev_from_mean"]  = (df["TransactionAmt"] - grp.transform("mean")).astype("float32")

    logger.info("Customer features added.")
    return df