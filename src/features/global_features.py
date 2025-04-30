"""
Create global, dataset-wide features: time, log amounts,
card1 statistics, and device/email fingerprints.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - Hour, day, day-of-week from TransactionDT
      - log-Amount
      - per-card1 mean amount & ratio
      - time difference between transactions per card
      - device‐type count per card
      - email‐domain root

    Args:
        df: Raw DataFrame with a TransactionDT and TransactionAmt.

    Returns:
        DataFrame enriched with global features.
    """
    df = df.copy()
    logger.info("Adding global features.")

    # a) parse DT
    dt = pd.to_datetime(df["TransactionDT"], origin="2017-12-01", unit="s")
    df["hour"] = dt.dt.hour.astype("uint8")
    df["day"]  = dt.dt.day.astype("uint8")
    df["dow"]  = dt.dt.dayofweek.astype("uint8")

    # b) log-amt
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"]).astype("float32")

    # c) card1 stats
    grp = df.groupby("card1")["TransactionAmt"]
    df["card1_amt_mean"]    = grp.transform("mean").astype("float32")
    df["amt_to_card1_mean"] = (df["TransactionAmt"] / (df["card1_amt_mean"]+1e-6)).astype("float32")

    # d) time delta
    df.sort_values(["card1","TransactionDT"], inplace=True)
    df["prev_DT"]   = df.groupby("card1")["TransactionDT"].shift(1)
    df["time_diff"] = (df["TransactionDT"] - df["prev_DT"]).fillna(0).astype("uint32")
    df.drop("prev_DT", axis=1, inplace=True)

    # e) device count
    df["device_count"] = df.groupby("card1")["DeviceType"].transform("nunique").astype("uint8")

    # f) email root
    for c in ("P_emaildomain", "R_emaildomain"):
        if c in df:
            df[f"{c}_root"] = df[c].str.split(".",1).str[0].fillna("unknown")

    logger.info("Global features added.")
    return df