"""
dodo.py

Defines pipeline tasks using doit (http://pydoit.org/) to automate:
  - data extraction & downcasting
  - feature engineering
  - model training (global & customer)
  - evaluation & metric export

Run with:
    $ doit           # runs default tasks in order
    $ doit list      # list available tasks
    $ doit task_name # run a specific task
"""

import os
from pathlib import Path

# Task parameters
RAW_ZIP   = Path("data/raw/transaction_data.csv.zip")
EXTRACTED = Path("data/processed/transaction_data.csv")
METRICS   = Path("artifacts/metrics.json")

def task_extract_data():
    """1. unzip raw CSV into processed folder (once)."""
    return {
        "file_dep": [RAW_ZIP],
        "targets": [EXTRACTED],
        "actions": [
            # unzip only if EXTRACTED missing or older
            (f"unzip -p {RAW_ZIP} > {EXTRACTED}",)
        ],
        "clean": True,
    }

def task_downcast():
    """2. downcast numeric columns to reduce memory."""
    return {
        "file_dep": [EXTRACTED],
        "targets": ["data/processed/transaction_data_downcast.csv"],
        "actions": [
            "python -c \"import src.data.loader as L; L.downcast_csv('data/processed/transaction_data.csv','data/processed/transaction_data_downcast.csv')\""
        ],
        "clean": True,
    }

def task_features():
    """3. run feature engineering for global + customer features."""
    return {
        "file_dep": ["data/processed/transaction_data_downcast.csv"],
        "targets": ["data/processed/features_global.csv", "data/processed/features_customer.csv"],
        "actions": [
            "python -c \"import src.features.global_features as G; G.build_global_features('data/processed/transaction_data_downcast.csv','data/processed/features_global.csv')\"",
            "python -c \"import src.features.customer_features as C; C.build_customer_features('data/processed/transaction_data_downcast.csv','data/processed/features_customer.csv')\"",
        ],
        "clean": True,
    }

def task_train_global():
    """4. train global VAE, save model and metrics."""
    return {
        "file_dep": ["data/processed/features_global.csv"],
        "targets": ["models/global_vae.pt", METRICS],
        "actions": [
            "python -m src.training.train_global --features data/processed/features_global.csv --out_model models/global_vae.pt --out_metrics artifacts/metrics.json"
        ],
        "clean": True,
    }

def task_train_customers():
    """5. train customer-specific VAEs & append to metrics."""
    return {
        "file_dep": ["data/processed/features_customer.csv", "models/global_vae.pt"],
        "targets": [METRICS],
        "actions": [
            "python -m src.training.train_customer --features data/processed/features_customer.csv --global_model models/global_vae.pt --append_metrics artifacts/metrics.json"
        ],
    }

def task_evaluate():
    """6. generate comparison plots & dashboard metrics."""
    return {
        "file_dep": [METRICS],
        "actions": [
            "python -m src.evaluation.metrics_to_plots --metrics artifacts/metrics.json --out_dir streamlit/assets"
        ],
        "targets": [],  # side-effect only
    }

def task_clean():
    """Cleanup all generated files."""
    return {
        "actions": ["rm -rf data/processed/* models/* artifacts/* streamlit/assets/*"],
        "verbosity": 2,
    }

# Default tasks to run in sequence:
DOIT_CONFIG = {
    "default_tasks": [
        "extract_data",
        "downcast",
        "features",
        "train_global",
        "train_customers",
        "evaluate",
    ]
}