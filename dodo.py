"""
Enhanced dodo.py for IEEE-CIS fraud detection project.

Defines an automated pipeline using doit (http://pydoit.org/) to:
- Download and extract IEEE-CIS dataset (train and test)
- Perform data preprocessing and analysis
- Generate global and customer-specific features
- Train global and customer-specific VAE models with transfer learning
- Evaluate models on test dataset
- Generate dashboard assets

Run with:
    $ doit           # runs default tasks in order
    $ doit list      # list available tasks
    $ doit task_name # run a specific task
"""

import os
import sys
from pathlib import Path

# Constants
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")
METRICS_DIR = Path(ARTIFACTS_DIR / "metrics")
EVAL_DIR = Path(ARTIFACTS_DIR / "evaluation")
DASHBOARD_DIR = Path("streamlit/assets")

# Ensure directories exist
for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR, METRICS_DIR, EVAL_DIR, DASHBOARD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset files (both train and test)
TRANSACTION_TRAIN_ZIP = RAW_DIR / "train_transaction.csv.zip"
TRANSACTION_TRAIN_CSV = RAW_DIR / "train_transaction.csv"
IDENTITY_TRAIN_CSV = RAW_DIR / "train_identity.csv"
TRANSACTION_TEST_CSV = RAW_DIR / "test_transaction.csv"
IDENTITY_TEST_CSV = RAW_DIR / "test_identity.csv"

# Processed files
TRANSACTION_TRAIN_DOWNCAST = PROCESSED_DIR / "train_transaction_downcast.csv"
TRANSACTION_TEST_DOWNCAST = PROCESSED_DIR / "test_transaction_downcast.csv"
MERGED_TRAIN_DOWNCAST = PROCESSED_DIR / "train_merged_downcast.csv"
MERGED_TEST_DOWNCAST = PROCESSED_DIR / "test_merged_downcast.csv"
GLOBAL_FEATURES_TRAIN = PROCESSED_DIR / "features_global_train.csv"
GLOBAL_FEATURES_TEST = PROCESSED_DIR / "features_global_test.csv"
CUSTOMER_FEATURES_TRAIN = PROCESSED_DIR / "features_customer_train.csv"
CUSTOMER_FEATURES_TEST = PROCESSED_DIR / "features_customer_test.csv"
FEATURE_DISTRIBUTIONS = PROCESSED_DIR / "feature_distributions.json"

# Model files
GLOBAL_VAE = MODELS_DIR / "global_vae.pt"
CUSTOMER_MODELS_DIR = MODELS_DIR / "customers"

# Metrics files
GLOBAL_METRICS = METRICS_DIR / "global_metrics.json"
CUSTOMER_METRICS = METRICS_DIR / "customer_metrics_summary.csv"
TEST_METRICS = EVAL_DIR / "test/global_metrics.json"

# Dashboard asset files
DASHBOARD_GLOBAL_ROC_PR = DASHBOARD_DIR / "global_roc_pr.png"
DASHBOARD_CUSTOMER_PERF = DASHBOARD_DIR / "customer_performance.png"
DASHBOARD_ERROR_DIST = DASHBOARD_DIR / "error_distribution.png"

def task_check_environment():
    """0. Check Python environment and dependencies."""
    return {
        "actions": [
            "python -c \"import torch; print('PyTorch version:', torch.__version__)\"",
            "python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"",
            "python -c \"import pandas as pd; print('Pandas version:', pd.__version__)\"",
            "python -c \"import sklearn; print('Scikit-learn version:', sklearn.__version__)\"",
            "python -c \"import pyro; print('Pyro-PPL version:', pyro.__version__)\""
        ],
        "verbosity": 2
    }

def task_download_data():
    """1. Download IEEE-CIS fraud detection dataset from Kaggle."""
    return {
        "actions": [
            f"python scripts/download_data.py --output {RAW_DIR}"
        ],
        "targets": [
            TRANSACTION_TRAIN_ZIP, 
            IDENTITY_TRAIN_CSV,
            TRANSACTION_TEST_CSV, 
            IDENTITY_TEST_CSV
        ],
        "uptodate": [True],  # Only run once
        "verbosity": 2,
    }

def task_extract_data():
    """2. Extract transaction data from zip (if needed)."""
    return {
        "file_dep": [TRANSACTION_TRAIN_ZIP],
        "targets": [TRANSACTION_TRAIN_CSV],
        "actions": [
            f"unzip -o -d {RAW_DIR} {TRANSACTION_TRAIN_ZIP}"
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_analyze_data():
    """3. Analyze and prepare data for modeling."""
    return {
        "file_dep": [
            TRANSACTION_TRAIN_CSV, 
            IDENTITY_TRAIN_CSV,
            TRANSACTION_TEST_CSV, 
            IDENTITY_TEST_CSV
        ],
        "targets": [
            ARTIFACTS_DIR / "data_analysis" / "data_analysis_report.md"
        ],
        "actions": [
            f"python scripts/data_preparation.py --data-dir {RAW_DIR} --output-dir {ARTIFACTS_DIR}/data_analysis"
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_downcast_train():
    """4a. Downcast train data numeric columns to reduce memory usage."""
    return {
        "file_dep": [TRANSACTION_TRAIN_CSV, IDENTITY_TRAIN_CSV],
        "targets": [TRANSACTION_TRAIN_DOWNCAST, MERGED_TRAIN_DOWNCAST],
        "actions": [
            f"python -c \"from fraud_detection.data.loader import downcast_csv, load_data; " + 
            f"downcast_csv('{TRANSACTION_TRAIN_CSV}', '{TRANSACTION_TRAIN_DOWNCAST}'); " +
            f"df = load_data('{RAW_DIR}', train=True); " +
            f"df.to_csv('{MERGED_TRAIN_DOWNCAST}', index=False)\""
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_downcast_test():
    """4b. Downcast test data numeric columns to reduce memory usage."""
    return {
        "file_dep": [TRANSACTION_TEST_CSV, IDENTITY_TEST_CSV],
        "targets": [TRANSACTION_TEST_DOWNCAST, MERGED_TEST_DOWNCAST],
        "actions": [
            f"python -c \"from fraud_detection.data.loader import downcast_csv, load_data; " + 
            f"downcast_csv('{TRANSACTION_TEST_CSV}', '{TRANSACTION_TEST_DOWNCAST}'); " +
            f"df = load_data('{RAW_DIR}', train=False); " +
            f"df.to_csv('{MERGED_TEST_DOWNCAST}', index=False)\""
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_global_features_train():
    """5a. Generate global features for train transactions."""
    return {
        "file_dep": [MERGED_TRAIN_DOWNCAST],
        "targets": [GLOBAL_FEATURES_TRAIN],
        "actions": [
            f"python -c \"from fraud_detection.features.global_features import build_global_features; " +
            f"build_global_features('{MERGED_TRAIN_DOWNCAST}', '{GLOBAL_FEATURES_TRAIN}')\""
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_global_features_test():
    """5b. Generate global features for test transactions."""
    return {
        "file_dep": [MERGED_TEST_DOWNCAST],
        "targets": [GLOBAL_FEATURES_TEST],
        "actions": [
            f"python -c \"from fraud_detection.features.global_features import build_global_features; " +
            f"build_global_features('{MERGED_TEST_DOWNCAST}', '{GLOBAL_FEATURES_TEST}')\""
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_customer_features_train():
    """6a. Generate customer-specific features for train transactions."""
    return {
        "file_dep": [MERGED_TRAIN_DOWNCAST],
        "targets": [CUSTOMER_FEATURES_TRAIN],
        "actions": [
            f"python -c \"from fraud_detection.features.customer_features import build_customer_features; " +
            f"build_customer_features('{MERGED_TRAIN_DOWNCAST}', '{CUSTOMER_FEATURES_TRAIN}')\""
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_customer_features_test():
    """6b. Generate customer-specific features for test transactions."""
    return {
        "file_dep": [MERGED_TEST_DOWNCAST],
        "targets": [CUSTOMER_FEATURES_TEST],
        "actions": [
            f"python -c \"from fraud_detection.features.customer_features import build_customer_features; " +
            f"build_customer_features('{MERGED_TEST_DOWNCAST}', '{CUSTOMER_FEATURES_TEST}')\""
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_feature_distributions():
    """7. Calculate feature distributions for transaction simulator."""
    return {
        "file_dep": [GLOBAL_FEATURES_TRAIN, CUSTOMER_FEATURES_TRAIN],
        "targets": [FEATURE_DISTRIBUTIONS],
        "actions": [
            f"python -c \"import pandas as pd; import numpy as np; import json; " +
            f"df = pd.read_csv('{GLOBAL_FEATURES_TRAIN}'); " +
            f"dist = {{}}; " +
            f"for col in df.select_dtypes(include=['number']).columns: " +
            f"    if col != 'isFraud': " +
            f"        legit = df[df['isFraud'] == 0][col]; " +
            f"        dist[col] = {{'min': float(legit.min()), 'max': float(legit.max()), " +
            f"                     'median': float(legit.median()), 'mean': float(legit.mean())}}; " +
            f"for col in df.select_dtypes(exclude=['number']).columns: " +
            f"    if col != 'isFraud': " +
            f"        vals = df[col].value_counts(normalize=True); " +
            f"        dist[col] = {{'values': vals.index.tolist(), 'frequencies': vals.values.tolist()}}; " +
            f"with open('{FEATURE_DISTRIBUTIONS}', 'w') as f: json.dump(dist, f, indent=2)\""
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_train_global_model():
    """8. Train global VAE model on legitimate train transactions."""
    return {
        "file_dep": [GLOBAL_FEATURES_TRAIN],
        "targets": [GLOBAL_VAE, GLOBAL_METRICS],
        "actions": [
            f"python -m fraud_detection.training.train_global " +
            f"--data_dir {PROCESSED_DIR} " +
            f"--input {GLOBAL_FEATURES_TRAIN} " +
            f"--out_model {GLOBAL_VAE} " +
            f"--out_dir {METRICS_DIR} " +
            f"--epochs 50 " +
            f"--batch_size 128 " +
            f"--beta 1.0 " +
            f"--fbeta 30.0"
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_train_customer_models():
    """9. Train customer-specific VAE models with transfer learning."""
    return {
        "file_dep": [CUSTOMER_FEATURES_TRAIN, GLOBAL_VAE],
        "targets": [CUSTOMER_METRICS],
        "actions": [
            f"python -m fraud_detection.training.train_customer " +
            f"--data_path {CUSTOMER_FEATURES_TRAIN} " +
            f"--global_model_path {GLOBAL_VAE} " +
            f"--model_dir {CUSTOMER_MODELS_DIR} " +
            f"--metrics_dir {METRICS_DIR} " +
            f"--min_transactions 100 " +
            f"--max_customers 20 " +  # Limit to 20 customers for demonstration
            f"--epochs 30 " +
            f"--batch_size 64 " +
            f"--beta 1.0"
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_evaluate_test():
    """10. Evaluate models on test data."""
    return {
        "file_dep": [
            GLOBAL_VAE, 
            GLOBAL_METRICS, 
            GLOBAL_FEATURES_TEST, 
            CUSTOMER_FEATURES_TEST
        ],
        "targets": [TEST_METRICS],
        "actions": [
            f"python scripts/model_evaluation.py " +
            f"--data-dir {RAW_DIR} " +
            f"--model-dir {MODELS_DIR} " +
            f"--output-dir {EVAL_DIR} " +
            f"--skip-train"  # Skip evaluation on train data since we already have those metrics
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_generate_predictions():
    """11. Generate predictions on test data."""
    return {
        "file_dep": [
            GLOBAL_VAE, 
            GLOBAL_METRICS, 
            GLOBAL_FEATURES_TEST, 
            CUSTOMER_FEATURES_TEST
        ],
        "targets": [ARTIFACTS_DIR / "predictions" / "test_predictions.csv"],
        "actions": [
            f"python scripts/generate_predictions.py " +
            f"--data-dir {RAW_DIR} " +
            f"--model-dir {MODELS_DIR} " +
            f"--output-dir {ARTIFACTS_DIR}/predictions"
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_generate_dashboard_assets():
    """12. Generate visualization assets for the Streamlit dashboard."""
    return {
        "file_dep": [GLOBAL_METRICS, CUSTOMER_METRICS, TEST_METRICS],
        "targets": [
            DASHBOARD_GLOBAL_ROC_PR,
            DASHBOARD_CUSTOMER_PERF,
            DASHBOARD_ERROR_DIST
        ],
        "actions": [
            f"python -c \"import json; import pandas as pd; import matplotlib.pyplot as plt; import seaborn as sns; " +
            # Load metrics
            f"with open('{GLOBAL_METRICS}', 'r') as f: global_metrics = json.load(f); " +
            f"with open('{TEST_METRICS}', 'r') as f: test_metrics = json.load(f); " +
            f"customer_metrics = pd.read_csv('{CUSTOMER_METRICS}'); " +
            
            # Plot ROC/PR curves
            f"global_roc = pd.read_csv('{METRICS_DIR}/global_roc.csv'); " +
            f"global_pr = pd.read_csv('{METRICS_DIR}/global_pr.csv'); " +
            f"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)); " +
            f"ax1.plot(global_roc['fpr'], global_roc['tpr'], label='Train'); " +
            f"ax1.set_title('Global Model ROC Curve'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); " +
            f"ax1.grid(True); " +
            f"# Add test ROC if available " +
            f"test_roc_path = '{EVAL_DIR}/test/curves/global_roc.csv'; " +
            f"if os.path.exists(test_roc_path): " +
            f"    test_roc = pd.read_csv(test_roc_path); " +
            f"    ax1.plot(test_roc['fpr'], test_roc['tpr'], label='Test'); " +
            f"    ax1.legend(); " +
            f"ax2.plot(global_pr['recall'], global_pr['precision'], label='Train'); " +
            f"ax2.set_title('Global Model PR Curve'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); " +
            f"ax2.grid(True); " +
            f"# Add test PR if available " +
            f"test_pr_path = '{EVAL_DIR}/test/curves/global_pr.csv'; " +
            f"if os.path.exists(test_pr_path): " +
            f"    test_pr = pd.read_csv(test_pr_path); " +
            f"    ax2.plot(test_pr['recall'], test_pr['precision'], label='Test'); " +
            f"    ax2.legend(); " +
            f"plt.tight_layout(); " +
            f"plt.savefig('{DASHBOARD_GLOBAL_ROC_PR}', dpi=300, bbox_inches='tight'); " +
            
            # Plot customer performance
            f"plt.figure(figsize=(12, 5)); " +
            f"ax1 = plt.subplot(1, 2, 1); " +
            f"sns.histplot(customer_metrics['recall'], kde=True, ax=ax1); " +
            f"ax1.set_title('Customer Models: Recall Distribution'); " +
            f"ax1.axvline(global_metrics['performance']['recall'], color='red', linestyle='--', label='Global Model'); " +
            f"ax1.legend(); " +
            f"ax2 = plt.subplot(1, 2, 2); " +
            f"sns.histplot(customer_metrics['specificity'], kde=True, ax=ax2); " +
            f"ax2.set_title('Customer Models: Specificity Distribution'); " +
            f"ax2.axvline(global_metrics['performance']['specificity'], color='red', linestyle='--', label='Global Model'); " +
            f"ax2.legend(); " +
            f"plt.tight_layout(); " +
            f"plt.savefig('{DASHBOARD_CUSTOMER_PERF}', dpi=300, bbox_inches='tight'); " +
            
            # Plot error distribution
            f"error_dist_path = '{EVAL_DIR}/test/curves/global_error_dist.png'; " +
            f"if os.path.exists(error_dist_path): " +
            f"    import shutil; " +
            f"    shutil.copy(error_dist_path, '{DASHBOARD_ERROR_DIST}'); " +
            f"\""
        ],
        "verbosity": 2,
        "clean": True,
    }

def task_run_dashboard():
    """13. Launch the Streamlit dashboard."""
    return {
        "file_dep": [
            DASHBOARD_GLOBAL_ROC_PR,
            DASHBOARD_CUSTOMER_PERF,
            DASHBOARD_ERROR_DIST
        ],
        "actions": [
            f"cd streamlit && streamlit run app.py"
        ],
        "verbosity": 2,
        "uptodate": [lambda task, values: False],  # Always run this task when requested
    }

def task_clean_processed():
    """Remove processed data files."""
    return {
        "actions": [
            f"rm -rf {PROCESSED_DIR}/*"
        ],
        "verbosity": 2,
    }

def task_clean_models():
    """Remove trained model files."""
    return {
        "actions": [
            f"rm -rf {MODELS_DIR}/*"
        ],
        "verbosity": 2,
    }

def task_clean_artifacts():
    """Remove metrics and visualization artifacts."""
    return {
        "actions": [
            f"rm -rf {ARTIFACTS_DIR}/*",
            f"rm -rf {DASHBOARD_DIR}/*"
        ],
        "verbosity": 2,
    }

def task_clean_all():
    """Clean everything except raw data."""
    return {
        "actions": [
            f"rm -rf {PROCESSED_DIR}/*",
            f"rm -rf {MODELS_DIR}/*",
            f"rm -rf {ARTIFACTS_DIR}/*",
            f"rm -rf {DASHBOARD_DIR}/*"
        ],
        "verbosity": 2,
    }

# Default tasks to run in sequence
DOIT_CONFIG = {
    "default_tasks": [
        "check_environment",
        "download_data", 
        "extract_data",
        "analyze_data",
        "downcast_train",
        "downcast_test",
        "global_features_train",
        "global_features_test",
        "customer_features_train",
        "customer_features_test",
        "feature_distributions",
        "train_global_model",
        "train_customer_models",
        "evaluate_test",
        "generate_predictions",
        "generate_dashboard_assets"
    ],
    "continue": True,  # Continue on task failure
}