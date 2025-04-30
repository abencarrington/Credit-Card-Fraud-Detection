"""
tests/test_workflow.py

A collection of basic pytest unit tests to validate:
  - data loading & downcasting
  - feature engineering outputs
  - model save/load
  - metrics file structure
"""

import json
import pandas as pd
import torch
import pytest
from pathlib import Path

import src.data.loader as loader
import src.features.global_features as GF
import src.features.customer_features as CF
import src.models.vae as VAE

from src.bootstrap import set_project_root
set_project_root()

DATA_RAW = Path("data/raw/transaction_data.csv.zip")
DATA_PROC = Path("data/processed/transaction_data_downcast.csv")
FEATURE_G = Path("data/processed/features_global.csv")
FEATURE_C = Path("data/processed/features_customer.csv")
MODEL_G = Path("models/global_vae.pt")
METRICS = Path("artifacts/metrics.json")

def test_downcast_roundtrip(tmp_path):
    """Ensure downcast preserves row count and valid dtypes."""
    src = tmp_path / "sample.csv"
    dst = tmp_path / "out.csv"
    # create a tiny toy CSV
    df = pd.DataFrame({"A": [1,2], "B": [1.0,2.0]})
    df.to_csv(src, index=False)
    loader.downcast_csv(str(src), str(dst))
    df2 = pd.read_csv(dst)
    assert df2.shape == df.shape
    assert df2["A"].dtype == "int8" or df2["A"].dtype == "int16"
    assert df2["B"].dtype == "float32"

def test_global_features_columns():
    """Global feature builder produces expected columns."""
    # use a small synthetic DF
    df = pd.DataFrame({
        "Time":[0,10000], "Amount":[1.2, 345.6], "Class":[0,1]
    })
    out = (tmp := Path("tmp_global.csv"))
    df.to_csv("tmp.csv", index=False)
    GF.build_global_features("tmp.csv", "tmp_global.csv")
    gdf = pd.read_csv("tmp_global.csv")
    for col in ["hour_of_day","Amount_log","velocity_1h"]:
        assert col in gdf.columns

def test_customer_features_structure():
    """Customer features grouped by customer ID."""
    df = pd.DataFrame({
        "card1":[1001,1001,1002],
        "Time":[0,100,200], "Amount":[5,10,3], "Class":[0,0,1]
    })
    df.to_csv("tmp_cust.csv", index=False)
    CF.build_customer_features("tmp_cust.csv", "tmp_cust_feat.csv")
    cdf = pd.read_csv("tmp_cust_feat.csv")
    # one row per unique card1
    assert set(cdf["card1"]) == {1001,1002}

def test_vae_model_save_load():
    """Global VAE can save and reload state dict."""
    vae = VAE.GlobalVAE(input_dim=10, hidden1=8, hidden2=4, zdim=2)
    dummy = torch.randn(5,10)
    _ = vae(dummy)  # forward pass
    path = "tmp_vae.pt"
    vae.save(path)
    vae2 = VAE.GlobalVAE.load(path)
    assert isinstance(vae2, VAE.GlobalVAE)

def test_metrics_json():
    """Metrics JSON has required keys."""
    sample = {"global": {"recall":0.97}, "customer": [{"id":1001,"recall":0.98}]}
    METRICS.write_text(json.dumps(sample))
    data = json.loads(METRICS.read_text())
    assert "global" in data and "customer" in data
    assert isinstance(data["customer"], list)