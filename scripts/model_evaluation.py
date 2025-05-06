"""
Script for comprehensive evaluation of VAE models on both train and test datasets.

This script:
1. Evaluates global and customer-specific VAE models
2. Computes detailed performance metrics
3. Generates comparative visualizations
4. Analyzes feature importance with SHAP values

Usage:
    python scripts/model_evaluation.py [--data-dir DATA_DIR] [--model-dir MODEL_DIR] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import shap
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, classification_report, auc
)

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

def calculate_metrics(y_true, y_pred, scores=None, beta=1.0):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        scores: Raw anomaly scores (optional)
        beta: Beta value for F-beta score
        
    Returns:
        Dictionary of metrics
    """
    # Basic classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=beta)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        f"f{beta}_score": float(fbeta),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    
    # AUC metrics if scores are provided
    if scores is not None:
        auroc = roc_auc_score(y_true, scores)
        avg_precision = average_precision_score(y_true, scores)
        
        metrics["roc_auc"] = float(auroc)
        metrics["avg_precision"] = float(avg_precision)
    
    return metrics

def evaluate_model(
    model,
    data,
    threshold,
    device,
    batch_size=1000,
    customer_id=None
):
    """
    Evaluate model on given data.
    
    Args:
        model: VAE model to evaluate
        data: DataFrame to evaluate on
        threshold: Threshold for anomaly detection
        device: Torch device
        batch_size: Batch size for evaluation
        customer_id: Customer ID for filtering (if needed)
        
    Returns:
        Metrics, predictions, scores
    """
    # Filter by customer if specified
    if customer_id is not None:
        data = data[data["card1"] == customer_id].copy()
        
        # Skip if no data for this customer
        if len(data) == 0:
            logger.warning(f"No data found for customer {customer_id}")
            return None, None, None
    
    # Select numeric features
    X = data.select_dtypes(include="number")
    
    # Check if target column exists
    has_target = "isFraud" in data.columns
    y_true = data["isFraud"].values if has_target else None
    
    # Convert to numpy array
    X_np = X.values.astype(np.float32)
    
    # Compute reconstruction errors
    scores = []
    
    with torch.no_grad():
        model.eval()
        # Process in batches
        for i in range(0, len(X_np), batch_size):
            end = min(i + batch_size, len(X_np))
            batch = torch.tensor(X_np[i:end], device=device)
            batch_scores = model.compute_reconstruction_error(batch).cpu().numpy()
            scores.extend(batch_scores)
    
    # Convert to numpy array
    scores = np.array(scores)
    
    # Apply threshold
    predictions = (scores >= threshold).astype(int)
    
    # Calculate metrics if target exists
    metrics = None
    if has_target:
        metrics = calculate_metrics(y_true, predictions, scores)
    
    return metrics, predictions, scores

def compute_and_plot_curves(y_true, scores, output_path, model_name="model"):
    """
    Compute and save ROC and PR curves.
    
    Args:
        y_true: True labels
        scores: Anomaly scores
        output_path: Directory to save plots
        model_name: Name for plot files
    
    Returns:
        AUC metrics
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    # Save curves to CSV
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        os.path.join(output_path, f"{model_name}_roc.csv"), index=False
    )
    
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
        os.path.join(output_path, f"{model_name}_pr.csv"), index=False
    )
    
    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, f"{model_name}_roc.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot and save PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='red', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, f"{model_name}_pr.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }

def plot_error_distribution(normal_errors, fraud_errors, threshold, output_path, model_name="model"):
    """
    Plot histogram of reconstruction errors for normal and fraudulent transactions.
    
    Args:
        normal_errors: Errors for normal transactions
        fraud_errors: Errors for fraudulent transactions
        threshold: Classification threshold
        output_path: Directory to save plot
        model_name: Name for plot file
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal', density=True, color='green')
    plt.hist(fraud_errors, bins=50, alpha=0.5, label='Fraud', density=True, color='red')
    
    # Plot threshold
    plt.axvline(x=threshold, color='blue', linestyle='-', linewidth=2, 
                label=f'Threshold: {threshold:.2f}')
    
    # Plot percentiles of normal distribution
    percentiles = [90, 95, 97.5, 99]
    for p in percentiles:
        p_threshold = np.percentile(normal_errors, p)
        plt.axvline(x=p_threshold, color='gray', linestyle='--', alpha=0.7,
                    label=f'{p}th Percentile: {p_threshold:.2f}')
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title(f'{model_name} Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(output_path, f"{model_name}_error_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()

def compute_feature_importance(model, data, device, output_path, model_name="model", n_background=200, n_examples=100):
    """
    Compute and visualize feature importance using SHAP values.
    
    Args:
        model: VAE model
        data: Input data
        device: Torch device
        output_path: Directory to save plot
        model_name: Name for plot file
        n_background: Number of background samples for SHAP
        n_examples: Number of examples to explain
    """
    try:
        # Get feature names
        feature_names = data.select_dtypes(include="number").columns.tolist()
        
        # Select only numeric features
        X = data.select_dtypes(include="number").values.astype(np.float32)
        
        # Get background data (legitimate transactions)
        if "isFraud" in data.columns:
            background_data = X[data["isFraud"] == 0]
            if len(background_data) > n_background:
                indices = np.random.choice(len(background_data), n_background, replace=False)
                background_data = background_data[indices]
        else:
            # If no labels, use random sample
            indices = np.random.choice(len(X), n_background, replace=False)
            background_data = X[indices]
        
        # Get examples to explain
        if len(X) > n_examples:
            indices = np.random.choice(len(X), n_examples, replace=False)
            examples = X[indices]
        else:
            examples = X
        
        # Convert to torch tensors
        background_tensor = torch.tensor(background_data, device=device)
        examples_tensor = torch.tensor(examples, device=device)
        
        # Create DeepExplainer
        def model_fn(x):
            return model.reconstruct(x)
        
        explainer = shap.DeepExplainer(model_fn, background_tensor)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(examples_tensor)
        
        # Create and save summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, examples, feature_names=feature_names, show=False
        )
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{model_name}_shap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save top features
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance
        }).sort_values("importance", ascending=False)
        
        importance_df.to_csv(os.path.join(output_path, f"{model_name}_feature_importance.csv"), index=False)
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error computing feature importance: {str(e)}")
        return None

def evaluate_models_on_dataset(
    dataset_name,
    data,
    global_model,
    global_threshold,
    customer_models,
    customer_thresholds,
    output_dir,
    device,
    batch_size=1000
):
    """
    Evaluate models on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (train/test)
        data: DataFrame with features and optionally labels
        global_model: Global VAE model
        global_threshold: Threshold for global model
        customer_models: Dictionary of customer VAE models
        customer_thresholds: Dictionary of thresholds for customer models
        output_dir: Directory to save results
        device: Torch device
        batch_size: Batch size for evaluation
        
    Returns:
        Summary metrics
    """
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if dataset has labels
    has_labels = "isFraud" in data.columns
    
    if not has_labels:
        logger.warning(f"{dataset_name} dataset does not have fraud labels. Evaluation will be limited.")
    
    # Evaluate global model
    logger.info(f"Evaluating global model on {dataset_name} dataset...")
    global_metrics, global_preds, global_scores = evaluate_model(
        global_model, data, global_threshold, device, batch_size
    )
    
    # Save global metrics
    if global_metrics:
        global_metrics_path = os.path.join(dataset_dir, "global_metrics.json")
        with open(global_metrics_path, "w") as f:
            json.dump(global_metrics, f, indent=2)
    
    # If we have labels, generate performance curves and error distributions
    if has_labels:
        # Compute curves
        curves_dir = os.path.join(dataset_dir, "curves")
        y_true = data["isFraud"].values
        
        # Global model curves
        logger.info(f"Computing performance curves for global model...")
        compute_and_plot_curves(y_true, global_scores, curves_dir, "global")
        
        # Error distribution
        normal_errors = global_scores[y_true == 0]
        fraud_errors = global_scores[y_true == 1]
        
        logger.info(f"Plotting error distribution for global model...")
        plot_error_distribution(
            normal_errors, fraud_errors, global_threshold, 
            curves_dir, "global"
        )
        
        # Feature importance
        logger.info(f"Computing feature importance for global model...")
        compute_feature_importance(
            global_model, data, device, 
            dataset_dir, "global"
        )
    
    # Evaluate customer models (if available)
    if customer_models:
        logger.info(f"Evaluating {len(customer_models)} customer models...")
        
        customer_metrics = []
        customer_results = {
            "scores": [],
            "predictions": [],
            "customer_ids": []
        }
        
        for customer_id, customer_model in tqdm(customer_models.items(), desc=f"Evaluating customers"):
            # Get threshold for this customer
            threshold = customer_thresholds.get(customer_id, global_threshold)
            
            # Filter data for this customer
            customer_data = data[data["card1"] == customer_id]
            
            if len(customer_data) == 0:
                logger.warning(f"No {dataset_name} data found for customer {customer_id}")
                continue
            
            # Evaluate model
            metrics, preds, scores = evaluate_model(
                customer_model, customer_data, threshold, device, batch_size
            )
            
            if metrics:
                metrics["customer_id"] = customer_id
                metrics["transaction_count"] = len(customer_data)
                customer_metrics.append(metrics)
            
            # Store results
            for i, idx in enumerate(customer_data.index):
                customer_results["scores"].append(scores[i])
                customer_results["predictions"].append(preds[i])
                customer_results["customer_ids"].append(customer_id)
        
        # Save customer metrics
        if customer_metrics:
            # Convert to DataFrame
            metrics_df = pd.DataFrame(customer_metrics)
            
            # Save to CSV
            metrics_df.to_csv(os.path.join(dataset_dir, "customer_metrics.csv"), index=False)
            
            # Compute summary statistics
            summary = metrics_df.describe().T
            summary.to_csv(os.path.join(dataset_dir, "customer_metrics_summary.csv"))
            
            # If we have labels, create comparison visualizations
            if has_labels and len(metrics_df) > 1:
                # Plot distribution of key metrics
                plt.figure(figsize=(15, 10))
                
                metrics_to_plot = ["recall", "specificity", "precision", "roc_auc"]
                for i, metric in enumerate(metrics_to_plot):
                    if metric in metrics_df.columns:
                        plt.subplot(2, 2, i+1)
                        sns.histplot(metrics_df[metric], kde=True)
                        plt.axvline(x=global_metrics[metric], color='red', linestyle='--',
                                   label=f'Global Model: {global_metrics[metric]:.3f}')
                        plt.title(f'Distribution of {metric.capitalize()}')
                        plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(dataset_dir, "customer_metrics_distribution.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
        # Create combined customer prediction scores
        combined_customer_df = pd.DataFrame({
            "customer_id": customer_results["customer_ids"],
            "score": customer_results["scores"],
            "prediction": customer_results["predictions"]
        })
        
        # Combine with original data indices
        customer_indices = np.concatenate([
            data[data["card1"] == cid].index.values
            for cid in customer_results["customer_ids"]
        ])
        
        combined_customer_df["original_index"] = customer_indices
        
        # Sort by original index
        combined_customer_df = combined_customer_df.sort_values("original_index")
        
        # Save combined predictions
        combined_customer_df.to_csv(os.path.join(dataset_dir, "customer_predictions.csv"), index=False)
    
    # Return summary of results
    summary = {
        "dataset": dataset_name,
        "transaction_count": len(data),
        "global_model": global_metrics if global_metrics else {},
        "customer_models_count": len(customer_metrics) if customer_models else 0
    }
    
    if has_labels:
        summary["fraud_count"] = int(data["isFraud"].sum())
        summary["fraud_rate"] = float(data["isFraud"].mean())
    
    return summary

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate VAE fraud detection models")
    parser.add_argument("--data-dir", default="./data/raw", help="Directory containing dataset")
    parser.add_argument("--model-dir", default="./models", help="Directory containing trained models")
    parser.add_argument("--output-dir", default="./artifacts/evaluation", help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for evaluation")
    parser.add_argument("--max-customers", type=int, default=None, help="Maximum number of customers to evaluate")
    parser.add_argument("--skip-train", action="store_true", help="Skip evaluation on training data")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for evaluation")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load train data (if needed)
    train_data = None
    if not args.skip_train:
        logger.info("Loading and preprocessing train data...")
        train_transaction = load_data(args.data_dir, train=True)
        train_with_global = add_global_features(train_transaction)
        train_with_all = add_customer_features(train_with_global)
        train_data = preprocess_data(train_with_all)
        logger.info(f"Train data ready: {train_data.shape}")
    
    # Load test data
    logger.info("Loading and preprocessing test data...")
    test_transaction = load_data(args.data_dir, train=False)
    test_with_global = add_global_features(test_transaction)
    test_with_all = add_customer_features(test_with_global)
    test_data = preprocess_data(test_with_all)
    logger.info(f"Test data ready: {test_data.shape}")
    
    # Get input dimension (from test data since it's always loaded)
    input_dim = test_data.select_dtypes(include="number").shape[1]
    
    # Load global model
    global_model_path = os.path.join(args.model_dir, "global_vae.pt")
    global_model = load_model(global_model_path, input_dim)
    
    if global_model is None:
        logger.error("Failed to load global model. Exiting.")
        sys.exit(1)
    
    # Move model to the correct device
    global_model.to(device)
    
    # Load threshold
    threshold_path = os.path.join("artifacts/metrics", "global_metrics.json")
    global_threshold = load_threshold(threshold_path)
    
    # Load customer models
    customer_models = {}
    customer_thresholds = {}
    
    customer_model_dir = os.path.join(args.model_dir, "customers")
    if os.path.exists(customer_model_dir):
        logger.info("Loading customer models...")
        
        # Get all customer model files
        model_files = list(Path(customer_model_dir).glob("customer_*.pt"))
        
        # Limit number of customers if specified
        if args.max_customers is not None and args.max_customers < len(model_files):
            logger.info(f"Limiting to {args.max_customers} customer models")
            model_files = model_files[:args.max_customers]
        
        # Load each customer model
        for model_file in tqdm(model_files, desc="Loading customer models"):
            # Extract customer ID from filename
            try:
                customer_id = int(model_file.stem.split("_")[1])
            except ValueError:
                logger.warning(f"Could not extract customer ID from {model_file}")
                continue
            
            # Load customer model
            customer_model = load_model(
                str(model_file), input_dim, model_type="customer", customer_id=customer_id
            )
            
            if customer_model is not None:
                # Move model to the correct device
                customer_model.to(device)
                customer_models[customer_id] = customer_model
                
                # Load customer threshold
                metrics_path = os.path.join(
                    "artifacts/metrics/customers", f"customer_{customer_id}_metrics.json"
                )
                if os.path.exists(metrics_path):
                    customer_thresholds[customer_id] = load_threshold(metrics_path, global_threshold)
                else:
                    customer_thresholds[customer_id] = global_threshold
        
        logger.info(f"Loaded {len(customer_models)} customer models")
    
    # Evaluate on train data
    train_summary = None
    if train_data is not None:
        logger.info("Evaluating on train data...")
        train_summary = evaluate_models_on_dataset(
            "train",
            train_data,
            global_model,
            global_threshold,
            customer_models,
            customer_thresholds,
            args.output_dir,
            device,
            args.batch_size
        )
    
    # Evaluate on test data
    logger.info("Evaluating on test data...")
    test_summary = evaluate_models_on_dataset(
        "test",
        test_data,
        global_model,
        global_threshold,
        customer_models,
        customer_thresholds,
        args.output_dir,
        device,
        args.batch_size
    )
    
    # Combine summaries into overall report
    overall_summary = {
        "train": train_summary if train_summary else "skipped",
        "test": test_summary
    }
    
    # Save overall summary
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
    
    # Print key metrics
    if "global_model" in test_summary and test_summary["global_model"]:
        global_metrics = test_summary["global_model"]
        logger.info(f"Global model test metrics:")
        logger.info(f"  ROC AUC: {global_metrics.get('roc_auc', 'N/A')}")
        logger.info(f"  Recall: {global_metrics.get('recall', 'N/A')}")
        logger.info(f"  Specificity: {global_metrics.get('specificity', 'N/A')}")

if __name__ == "__main__":
    main()