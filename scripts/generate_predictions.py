"""
Script to generate predictions using trained models on the IEEE-CIS test dataset.

This script:
1. Loads the trained VAE models (global and customer-specific)
2. Preprocesses the test dataset
3. Generates anomaly scores for each transaction
4. Evaluates performance metrics

Usage:
    python scripts/generate_predictions.py [--data-dir DATA_DIR] [--model-dir MODEL_DIR] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

from src.data.loader import load_data
from src.data.processor import preprocess_data
from src.features.global_features import add_global_features
from src.features.customer_features import add_customer_features
from src.models.vae import GlobalVAE, CustomerVAE
from src.bootstrap import set_project_root

# Set project root to resolve imports correctly
set_project_root()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path, input_dim, model_type="global", customer_id=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        input_dim: Input dimension for the model
        model_type: Type of model ('global' or 'customer')
        customer_id: Customer ID for customer-specific models
        
    Returns:
        Loaded model
    """
    try:
        if model_type == "global":
            model = GlobalVAE.load(model_path, input_dim=input_dim)
        else:
            model = CustomerVAE.load(model_path, input_dim=input_dim, customer_id=customer_id)
        
        model.eval()  # Set to evaluation mode
        logger.info(f"Loaded {model_type} model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def load_threshold(metrics_path, default=50.0):
    """
    Load the optimal threshold from metrics file.
    
    Args:
        metrics_path: Path to metrics JSON file
        default: Default threshold if metrics file not found
        
    Returns:
        Threshold value
    """
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        threshold = metrics.get("performance", {}).get("threshold", default)
        logger.info(f"Loaded threshold {threshold} from {metrics_path}")
        return threshold
    except Exception as e:
        logger.warning(f"Error loading threshold from {metrics_path}: {str(e)}")
        logger.warning(f"Using default threshold of {default}")
        return default

def generate_predictions(
    data_dir="./data/raw",
    model_dir="./models",
    output_dir="./artifacts/predictions",
    batch_size=1000,
    device=None
):
    """
    Generate predictions on the test dataset.
    
    Args:
        data_dir: Directory containing the dataset
        model_dir: Directory containing trained models
        output_dir: Directory to save predictions and metrics
        batch_size: Batch size for prediction
        device: Torch device (CPU/GPU)
        
    Returns:
        DataFrame with predictions
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    logger.info("Loading test data...")
    test_transaction = load_data(data_dir, train=False)
    
    # Keep a copy of original IDs
    original_ids = test_transaction["TransactionID"].copy()
    
    # Add features
    logger.info("Adding global features...")
    test_with_global = add_global_features(test_transaction)
    
    logger.info("Adding customer features...")
    test_with_all = add_customer_features(test_with_global)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    test_processed = preprocess_data(test_with_all)
    
    # Handle non-numeric columns
    X_test = test_processed.select_dtypes(include="number")
    
    # Get input dimension
    input_dim = X_test.shape[1]
    logger.info(f"Test data ready with {X_test.shape[0]} transactions and {input_dim} features")
    
    # Load global model
    global_model_path = os.path.join(model_dir, "global_vae.pt")
    global_model = load_model(global_model_path, input_dim)
    
    if global_model is None:
        logger.error("Failed to load global model. Exiting.")
        return None
    
    # Load threshold
    threshold_path = os.path.join("artifacts/metrics", "global_metrics.json")
    threshold = load_threshold(threshold_path)
    
    # Load customer models
    customer_models = {}
    customer_thresholds = {}
    
    customer_model_dir = os.path.join(model_dir, "customers")
    if os.path.exists(customer_model_dir):
        logger.info("Loading customer models...")
        
        # Get unique customers in test data
        test_customers = test_processed["card1"].unique()
        
        # Load models for customers that exist in test data
        for customer_id in tqdm(test_customers, desc="Loading customer models"):
            model_path = os.path.join(customer_model_dir, f"customer_{customer_id}.pt")
            if os.path.exists(model_path):
                customer_model = load_model(
                    model_path, input_dim, model_type="customer", customer_id=customer_id
                )
                if customer_model is not None:
                    customer_models[customer_id] = customer_model
                    
                    # Load customer threshold
                    metrics_path = os.path.join(
                        "artifacts/metrics/customers", f"customer_{customer_id}_metrics.json"
                    )
                    if os.path.exists(metrics_path):
                        customer_thresholds[customer_id] = load_threshold(metrics_path, threshold)
                    else:
                        customer_thresholds[customer_id] = threshold
        
        logger.info(f"Loaded {len(customer_models)} customer models")
    
    # Convert test data to tensor
    X_test_np = X_test.values.astype(np.float32)
    
    # Generate predictions in batches
    logger.info("Generating predictions...")
    
    global_scores = []
    customer_scores = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        for i in range(0, len(X_test_np), batch_size):
            end = min(i + batch_size, len(X_test_np))
            batch = torch.tensor(X_test_np[i:end], device=device)
            
            # Global model predictions
            global_model.eval()
            global_errors = global_model.compute_reconstruction_error(batch).cpu().numpy()
            global_scores.extend(global_errors)
            
            # Customer model predictions (if available)
            batch_customer_scores = np.zeros(end - i)
            batch_card_ids = test_processed["card1"].iloc[i:end].values
            
            for j, card_id in enumerate(batch_card_ids):
                if card_id in customer_models:
                    customer_models[card_id].eval()
                    # Extract customer-specific features for this transaction
                    customer_sample = batch[j:j+1]
                    customer_error = customer_models[card_id].compute_reconstruction_error(
                        customer_sample
                    ).cpu().item()
                    batch_customer_scores[j] = customer_error
                else:
                    # Fall back to global model for customers without specific model
                    batch_customer_scores[j] = global_errors[j]
            
            customer_scores.extend(batch_customer_scores)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        "TransactionID": original_ids,
        "global_score": global_scores,
        "customer_score": customer_scores
    })
    
    # Add predictions based on thresholds
    results["global_prediction"] = (results["global_score"] >= threshold).astype(int)
    
    # For customer predictions, use customer-specific thresholds
    results["customer_prediction"] = np.zeros(len(results), dtype=int)
    for customer_id, cust_threshold in customer_thresholds.items():
        mask = test_processed["card1"] == customer_id
        results.loc[mask, "customer_prediction"] = (
            results.loc[mask, "customer_score"] >= cust_threshold
        ).astype(int)
    
    # Save predictions
    output_path = os.path.join(output_dir, "test_predictions.csv")
    results.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    # If test labels are available (for validation), compute metrics
    if "isFraud" in test_transaction.columns:
        y_true = test_transaction["isFraud"].values
        
        # Compute global model metrics
        global_auc = roc_auc_score(y_true, results["global_score"])
        global_avg_precision = average_precision_score(y_true, results["global_score"])
        
        # Compute customer model metrics
        customer_auc = roc_auc_score(y_true, results["customer_score"])
        customer_avg_precision = average_precision_score(y_true, results["customer_score"])
        
        # Save metrics
        metrics = {
            "global_model": {
                "roc_auc": float(global_auc),
                "avg_precision": float(global_avg_precision)
            },
            "customer_models": {
                "roc_auc": float(customer_auc),
                "avg_precision": float(customer_avg_precision)
            }
        }
        
        metrics_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Test metrics: Global AUC = {global_auc:.4f}, Customer AUC = {customer_auc:.4f}")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate predictions on test data")
    parser.add_argument("--data-dir", default="./data/raw", help="Directory containing dataset")
    parser.add_argument("--model-dir", default="./models", help="Directory containing trained models")
    parser.add_argument("--output-dir", default="./artifacts/predictions", help="Directory to save predictions")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for prediction")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for prediction")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Generate predictions
    generate_predictions(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device
    )

if __name__ == "__main__":
    main()