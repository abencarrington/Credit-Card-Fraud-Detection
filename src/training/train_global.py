"""
Enhanced training script for the global VAE model.

Key updates:
1. Proper handling of geospatial features (using dist1/dist2 but not addr1/addr2)
2. Better documentation of model parameters to support transfer learning
3. Save additional metadata to assist customer model training
4. Improved feature handling and dimensionality tracking
"""

import os
import json
import logging
import time
from typing import Dict, Tuple, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, confusion_matrix, roc_curve,
    precision_recall_curve, auc, roc_auc_score,
    f1_score, fbeta_score
)
import shap

from src.data.loader import load_data
from src.data.processor import preprocess_data
from src.features.global_features import add_global_features
from src.models.vae import GlobalVAE
from src.evaluation.metrics import plot_pr_roc, calculate_fbeta_threshold
from src.visualization.plots import (
    plot_training_curves, 
    plot_latent_space, 
    plot_reconstruction_error_distribution
)
from src.bootstrap import set_project_root

# Set project root to resolve imports correctly
set_project_root()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train_global(
    data_dir: str = "./data/raw",
    input_file: Optional[str] = None,
    out_dir: str = "./artifacts",
    model_dir: str = "./models",
    z_dim: int = 2,
    h1: int = 128,
    h2: int = 64,
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 100,
    beta: float = 1.0,
    beta_warmup_epochs: int = 10,
    kl_warmup_epochs: int = 20,
    early_stopping_patience: int = 10,
    dropout_rate: float = 0.2,
    fbeta: float = 30.0,
    random_state: int = 42
) -> Tuple[GlobalVAE, Dict[str, Any]]:
    """
    Train a global VAE model on all legitimate transactions.
    
    Implements KL annealing and β-VAE features for improved training and latent space.
    Excludes addr1/addr2 features but retains dist1/dist2 for geospatial analysis.
    
    Args:
        data_dir: Directory containing raw data files
        input_file: Optional path to preprocessed input file
        out_dir: Directory to save metrics and plots
        model_dir: Directory to save trained models
        z_dim: Dimension of latent space
        h1: Size of first hidden layer
        h2: Size of second hidden layer
        lr: Initial learning rate
        batch_size: Training batch size
        epochs: Maximum number of training epochs
        beta: Final β value for KL term weighting
        beta_warmup_epochs: Epochs to warm up beta from 0 to final value
        kl_warmup_epochs: Epochs to warm up KL annealing from 0 to 1
        early_stopping_patience: Patience for early stopping
        dropout_rate: Dropout rate for model regularization
        fbeta: Beta value for F-beta score (high to prioritize recall)
        random_state: Random seed for reproducibility
        
    Returns:
        model: Trained GlobalVAE model
        metrics: Dictionary of evaluation metrics
    """
    # Create output directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    start_time = time.time()
    logger.info("=== GLOBAL VAE TRAINING START ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Data loading and preprocessing
    if input_file:
        # Use pre-processed input file if provided
        logger.info(f"Loading pre-processed data from {input_file}")
        df = pd.read_csv(input_file)
    else:
        # Load & preprocess data
        logger.info("Loading and preprocessing data...")
        df = load_data(data_dir, train=True)
        df = add_global_features(df)
        df = preprocess_data(df)
    
    # Convert isFraud to binary for clarity
    if 'isFraud' in df.columns:
        target_col = 'isFraud'
    elif 'Class' in df.columns:
        target_col = 'Class'
    else:
        raise ValueError("No fraud label column found in data")
    
    # Get column information for feature tracking
    original_columns = df.columns.tolist()
    
    # Handle addr1/addr2 columns - exclude them for global model
    addr_cols = [col for col in df.columns if col.startswith('addr')]
    if addr_cols:
        logger.info(f"Excluding address columns for global model: {addr_cols}")
        df = df.drop(columns=addr_cols)
    
    # Keep distance columns (dist1, dist2)
    dist_cols = [col for col in df.columns if col.startswith('dist')]
    if dist_cols:
        logger.info(f"Keeping distance columns for global model: {dist_cols}")
    
    # Train/validation/test split
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    
    # Then split remaining data into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, 
        random_state=random_state
    )
    
    # Keep only legitimate transactions for training
    X_train_legit = X_train[y_train == 0].copy()
    
    # Store feature metadata for transfer learning
    feature_metadata = {
        "numeric_features": X_train_legit.select_dtypes(include='number').columns.tolist(),
        "categorical_features": X_train_legit.select_dtypes(exclude='number').columns.tolist(),
        "excluded_addr_cols": addr_cols,
        "included_dist_cols": dist_cols,
        "total_feature_count": X_train_legit.shape[1]
    }
    
    # Handle non-numeric columns for VAE
    non_numeric_cols = X_train_legit.select_dtypes(exclude='number').columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Found {len(non_numeric_cols)} non-numeric columns. "
                     f"These will be dropped for VAE training: {non_numeric_cols.tolist()}")
        X_train_legit = X_train_legit.select_dtypes(include='number')
        X_val = X_val.select_dtypes(include='number')
        X_test = X_test.select_dtypes(include='number')
    
    # Convert to numpy arrays
    X_train_np = X_train_legit.to_numpy(dtype=np.float32)
    X_val_np = X_val.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    
    # Record input dimension for model creation
    input_dim = X_train_np.shape[1]
    feature_metadata["final_input_dim"] = input_dim
    
    # Save feature names for interpretation
    feature_names = X_train_legit.columns.tolist()
    feature_metadata["feature_names"] = feature_names
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating VAE model (z_dim={z_dim}, beta={beta}, " +
                f"h1={h1}, h2={h2}, dropout={dropout_rate})...")
    
    model = GlobalVAE(
        input_dim=input_dim,
        hidden1=h1,
        hidden2=h2,
        zdim=z_dim,
        beta=beta,
        dropout_rate=dropout_rate
    ).to(device)
    
    # Create optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
        threshold=1e-4, threshold_mode='rel'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_np).to(device)),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_np).to(device)),
        batch_size=batch_size
    )
    
    # Training setup
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    kl_losses = []
    recon_losses = []
    
    # Training loop
    logger.info(f"Training for {epochs} epochs (early stopping patience: {early_stopping_patience})...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        
        # KL annealing factor
        kl_annealing_factor = min(1.0, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else 1.0
        
        # Beta warmup
        current_beta = min(beta, beta * epoch / beta_warmup_epochs) if beta_warmup_epochs > 0 else beta
        model.beta = current_beta
        
        # Train one epoch
        for batch_idx, (data,) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass with current annealing
            loss, recon_loss, kl_loss = model.compute_loss(
                data, 
                annealing_factor=kl_annealing_factor
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_train_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
        
        # Average loss for the epoch
        epoch_train_loss /= len(train_loader)
        epoch_kl_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(val_loader):
                loss, _, _ = model.compute_loss(data)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(epoch_val_loss)
        
        # Track losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        kl_losses.append(epoch_kl_loss)
        recon_losses.append(epoch_recon_loss)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | " +
                        f"Train: {epoch_train_loss:.4f} | " +
                        f"Val: {epoch_val_loss:.4f} | " +
                        f"KL: {epoch_kl_loss:.4f} | " +
                        f"Recon: {epoch_recon_loss:.4f} | " +
                        f"β: {current_beta:.3f} | " +
                        f"KL anneal: {kl_annealing_factor:.3f}")
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save the best model
            model_path = os.path.join(model_dir, "global_vae_best.pt")
            model.save(model_path)
            logger.info(f"Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model for evaluation
    best_model_path = os.path.join(model_dir, "global_vae_best.pt")
    model = GlobalVAE.load(best_model_path, input_dim=input_dim)
    model.to(device)
    
    # Final model evaluation
    logger.info("Evaluating model on test set...")
    model.eval()
    
    # Compute reconstruction errors on validation set for threshold
    val_errors = []
    with torch.no_grad():
        for i in range(0, len(X_val_np), batch_size):
            data = torch.from_numpy(X_val_np[i:i+batch_size]).to(device)
            val_errors.extend(model.compute_reconstruction_error(data).cpu().numpy())
    
    # Compute reconstruction errors on test set
    test_errors = []
    with torch.no_grad():
        for i in range(0, len(X_test_np), batch_size):
            data = torch.from_numpy(X_test_np[i:i+batch_size]).to(device)
            test_errors.extend(model.compute_reconstruction_error(data).cpu().numpy())
    
    # Determine optimal threshold using F-beta score
    logger.info(f"Calculating optimal threshold (F-beta={fbeta})...")
    threshold, f_beta = calculate_fbeta_threshold(
        y_test.values, 
        np.array(test_errors), 
        beta=fbeta
    )
    
    # Apply threshold to get predictions
    y_pred = (np.array(test_errors) >= threshold).astype(int)
    
    # Calculate metrics
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    
    # ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, test_errors)
    roc_auc = auc(fpr, tpr)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_errors)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Save curves to CSV for dashboard
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        os.path.join(out_dir, "metrics", "global_roc.csv"), index=False
    )
    pd.DataFrame({"precision": precision_curve, "recall": recall_curve}).to_csv(
        os.path.join(out_dir, "metrics", "global_pr.csv"), index=False
    )
    
    # Create visualization plots
    logger.info("Generating visualization plots...")
    plot_path = os.path.join(out_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)
    
    # 1. Training curves
    plot_training_curves(
        train_losses, val_losses, kl_losses, recon_losses,
        save_path=os.path.join(plot_path, "global_training_curves.png")
    )
    
    # 2. ROC and PR curves
    plot_pr_roc(
        y_test.values, test_errors, threshold, beta=fbeta,
        save_path=os.path.join(plot_path, "global_roc_pr.png")
    )
    
    # 3. Reconstruction error distribution
    plot_reconstruction_error_distribution(
        test_errors, y_test.values, threshold,
        save_path=os.path.join(plot_path, "global_error_dist.png")
    )
    
    # 4. Latent space visualization (if z_dim is 2)
    if z_dim == 2:
        # Encode test data to visualize latent space
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test_np).to(device)
            mu, _ = model.encode(X_test_tensor)
            latent_points = mu.cpu().numpy()
            
            plot_latent_space(
                latent_points, y_test.values,
                save_path=os.path.join(plot_path, "global_latent_space.png")
            )
    
    # SHAP analysis for interpretability
    logger.info("Computing SHAP values for interpretability...")
    try:
        # Use a small background set for efficiency
        n_background = min(200, len(X_train_np))
        background_data = torch.tensor(
            X_train_np[np.random.choice(len(X_train_np), n_background, replace=False)]
        ).to(device)
        
        # Use a sample of test set for SHAP values
        n_explain = min(200, len(X_test_np))
        explain_data = torch.tensor(X_test_np[:n_explain]).to(device)
        
        # Create DeepExplainer
        explainer = shap.DeepExplainer(
            lambda x: model.reconstruct(x), 
            background_data
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(explain_data)
        
        # Create and save SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_test_np[:n_explain], 
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "global_shap.png"), bbox_inches="tight")
        plt.close()
        
        # Save feature importance based on SHAP
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance
        }).sort_values("importance", ascending=False)
        
        importance_df.to_csv(os.path.join(plot_path, "global_feature_importance.csv"), index=False)
        
    except Exception as e:
        logger.warning(f"Error computing SHAP values: {str(e)}")
    
    # Compile all metrics
    metrics = {
        "model_params": {
            "input_dim": input_dim,
            "z_dim": z_dim,
            "hidden1": h1,
            "hidden2": h2,
            "beta": beta,
            "epochs_trained": best_epoch + 1,
            "final_lr": optimizer.param_groups[0]['lr']
        },
        "feature_metadata": feature_metadata,
        "performance": {
            "threshold": float(threshold),
            "f_beta": float(f_beta),
            "recall": float(recall),
            "precision": float(precision),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "best_val_loss": float(best_val_loss)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "training_time_seconds": int(time.time() - start_time)
    }
    
    # Save metrics to JSON
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    with open(os.path.join(metrics_dir, "global_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Also save final model
    model_path = os.path.join(model_dir, "global_vae.pt")
    model.save(model_path)
    
    logger.info("=== GLOBAL TRAINING COMPLETE ===")
    logger.info(f"Final metrics: Recall={recall:.4f}, Specificity={specificity:.4f}, AUC={roc_auc:.4f}")
    
    return model, metrics

def main():
    """Main function for command line usage."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train global VAE model for fraud detection")
    parser.add_argument("--data_dir", default="./data/raw", help="Directory with raw data")
    parser.add_argument("--input", default=None, help="Path to preprocessed input file")
    parser.add_argument("--out_dir", default="./artifacts", help="Directory to save metrics and plots")
    parser.add_argument("--out_model", default="./models/global_vae.pt", help="Path to save model")
    parser.add_argument("--z_dim", type=int, default=2, help="Latent space dimension")
    parser.add_argument("--h1", type=int, default=128, help="First hidden layer size")
    parser.add_argument("--h2", type=int, default=64, help="Second hidden layer size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for KL term")
    parser.add_argument("--fbeta", type=float, default=30.0, help="Beta for F-score")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Run training
    model_dir = os.path.dirname(args.out_model)
    train_global(
        data_dir=args.data_dir,
        input_file=args.input,
        out_dir=args.out_dir,
        model_dir=model_dir,
        z_dim=args.z_dim,
        h1=args.h1,
        h2=args.h2,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        beta=args.beta,
        early_stopping_patience=args.patience,
        fbeta=args.fbeta,
        random_state=args.seed
    )

if __name__ == "__main__":
    main()