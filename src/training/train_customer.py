"""
Enhanced customer-specific VAE training module with improved transfer learning.

Key improvements:
1. More efficient transfer learning from global model
2. No redundant global model training
3. Better weight transfer between models
4. Focused training on customer-specific patterns
"""

import os
import logging
import json
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score, fbeta_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

from typing import Dict, List, Tuple, Optional, Any, Union

from src.data.loader import load_data
from src.data.processor import preprocess_data
from src.features.customer_features import add_customer_features
from src.models.vae import GlobalVAE, CustomerVAE
from src.evaluation.metrics import calculate_fbeta_threshold, calculate_performance_metrics

logger = logging.getLogger(__name__)

def train_customer_vae(
    customer_id: int,
    data: pd.DataFrame,
    global_model: GlobalVAE,
    device: torch.device = torch.device("cpu"),
    z_dim: int = 2,
    h1: int = 64,
    h2: int = 32,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 50,
    beta: float = 1.0,
    beta_warmup_epochs: int = 5,
    kl_warmup_epochs: int = 10,
    early_stopping_patience: int = 5,
    dropout_rate: float = 0.2,
    model_dir: str = "./models/customers",
    fbeta: float = 30.0,
    min_txns: int = 100,
    random_state: int = 42
) -> Tuple[Optional[CustomerVAE], Optional[Dict[str, Any]]]:
    """
    Train a customer-specific VAE model with improved transfer learning from the global model.
    
    Args:
        customer_id: Card ID of the customer
        data: Preprocessed DataFrame with all transactions
        global_model: Pre-trained global VAE for transfer learning
        device: Torch device (CPU or GPU)
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
        model_dir: Directory to save trained models
        fbeta: Beta value for F-beta score (high to prioritize recall)
        min_txns: Minimum number of transactions required for training
        random_state: Random seed for reproducibility
        
    Returns:
        model: Trained CustomerVAE model (or None if insufficient data)
        metrics: Dictionary of evaluation metrics (or None if training failed)
    """
    logger.info(f"Training customer VAE for customer ID {customer_id}")
    
    start_time = time.time()
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Filter data for this customer
    customer_data = data[data['card1'] == customer_id].copy()
    
    # Check if enough data
    if len(customer_data) < min_txns:
        logger.warning(f"Customer {customer_id} has only {len(customer_data)} transactions, " +
                     f"which is below minimum threshold of {min_txns}. Skipping.")
        return None, None
    
    # Check if fraud instances exist
    has_fraud = 'isFraud' in customer_data.columns and customer_data['isFraud'].sum() > 0
    logger.info(f"Customer {customer_id} has {len(customer_data)} transactions, " +
               f"including {customer_data['isFraud'].sum() if has_fraud else 0} fraudulent ones.")
    
    # Prepare data for VAE
    if 'isFraud' in customer_data.columns:
        X = customer_data.drop('isFraud', axis=1)
        y = customer_data['isFraud']
    else:
        X = customer_data
        y = None
    
    # Get categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Drop non-numeric columns for VAE training
    if len(cat_cols) > 0:
        X = X.drop(cat_cols, axis=1)
    
    # Split data
    if has_fraud:
        # Stratified split to preserve fraud ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=random_state
        )
        
        # For training, use only legitimate transactions
        X_train_legit = X_train[y_train == 0].to_numpy(dtype=np.float32)
    else:
        # Without fraud, use simple split
        X_train, X_test = train_test_split(
            X, test_size=0.25, random_state=random_state
        )
        
        # For testing, create dummy target
        y_test = np.zeros(len(X_test))
        
        # Use all training data
        X_train_legit = X_train.to_numpy(dtype=np.float32)
    
    # Convert test data to numpy
    X_test_np = X_test.to_numpy(dtype=np.float32)
    
    # Check if global model is compatible with input dimension
    input_dim = X_train_legit.shape[1]
    
    # Check if global model and customer data have compatible dimensions
    global_model_input_dim = global_model.encoder.fc1.weight.shape[1]
    
    if global_model_input_dim != input_dim:
        logger.warning(f"Global model input dimension ({global_model_input_dim}) " +
                     f"does not match customer data dimension ({input_dim}). " +
                     f"Will use partial weight transfer.")
    
    # Create new customer-specific model
    logger.info(f"Initializing customer model with transfer learning from global model")
    customer_model = CustomerVAE(
        input_dim=input_dim,
        customer_id=customer_id,
        hidden1=h1,
        hidden2=h2,
        zdim=z_dim,
        beta=beta,
        dropout_rate=dropout_rate
    ).to(device)
    
    # Transfer weights from global model to customer model
    with torch.no_grad():
        # Transfer encoder weights with dimension handling
        if global_model_input_dim == input_dim:
            # Direct transfer if dimensions match
            customer_model.encoder.fc1.weight.copy_(global_model.encoder.fc1.weight)
            customer_model.encoder.fc1.bias.copy_(global_model.encoder.fc1.bias)
        else:
            # Partial transfer for first layer if dimensions don't match
            min_dim = min(global_model_input_dim, input_dim)
            customer_model.encoder.fc1.weight[:, :min_dim].copy_(
                global_model.encoder.fc1.weight[:, :min_dim]
            )
            customer_model.encoder.fc1.bias.copy_(global_model.encoder.fc1.bias)
            
            # Initialize remaining weights with similar statistics
            if input_dim > global_model_input_dim:
                mean_weight = global_model.encoder.fc1.weight.mean().item()
                std_weight = global_model.encoder.fc1.weight.std().item()
                # Initialize new dimensions with similar distribution
                nn.init.normal_(
                    customer_model.encoder.fc1.weight[:, min_dim:],
                    mean=mean_weight, std=std_weight
                )
        
        # Transfer remaining encoder layers (dimensions should match)
        customer_model.encoder.fc2.weight.copy_(global_model.encoder.fc2.weight)
        customer_model.encoder.fc2.bias.copy_(global_model.encoder.fc2.bias)
        customer_model.encoder.fc_mu.weight.copy_(global_model.encoder.fc_mu.weight)
        customer_model.encoder.fc_mu.bias.copy_(global_model.encoder.fc_mu.bias)
        customer_model.encoder.fc_logvar.weight.copy_(global_model.encoder.fc_logvar.weight)
        customer_model.encoder.fc_logvar.bias.copy_(global_model.encoder.fc_logvar.bias)
        
        # Transfer decoder layers (except the final layer if dimensions differ)
        customer_model.decoder.fc1.weight.copy_(global_model.decoder.fc1.weight)
        customer_model.decoder.fc1.bias.copy_(global_model.decoder.fc1.bias)
        customer_model.decoder.fc2.weight.copy_(global_model.decoder.fc2.weight)
        customer_model.decoder.fc2.bias.copy_(global_model.decoder.fc2.bias)
        
        if global_model_input_dim == input_dim:
            # Direct transfer if dimensions match
            customer_model.decoder.fc_out.weight.copy_(global_model.decoder.fc_out.weight)
            customer_model.decoder.fc_out.bias.copy_(global_model.decoder.fc_out.bias)
            customer_model.decoder.fc_logvar.weight.copy_(global_model.decoder.fc_logvar.weight)
            customer_model.decoder.fc_logvar.bias.copy_(global_model.decoder.fc_logvar.bias)
        else:
            # Partial transfer for final layer if dimensions don't match
            min_dim = min(global_model_input_dim, input_dim)
            customer_model.decoder.fc_out.weight[:min_dim, :].copy_(
                global_model.decoder.fc_out.weight[:min_dim, :]
            )
            customer_model.decoder.fc_out.bias[:min_dim].copy_(
                global_model.decoder.fc_out.bias[:min_dim]
            )
            customer_model.decoder.fc_logvar.weight[:min_dim, :].copy_(
                global_model.decoder.fc_logvar.weight[:min_dim, :]
            )
            customer_model.decoder.fc_logvar.bias[:min_dim].copy_(
                global_model.decoder.fc_logvar.bias[:min_dim]
            )
            
            # Initialize remaining weights with similar statistics
            if input_dim > global_model_input_dim:
                mean_weight = global_model.decoder.fc_out.weight.mean().item()
                std_weight = global_model.decoder.fc_out.weight.std().item()
                # Initialize new dimensions with similar distribution
                nn.init.normal_(
                    customer_model.decoder.fc_out.weight[min_dim:, :],
                    mean=mean_weight, std=std_weight
                )
                nn.init.normal_(
                    customer_model.decoder.fc_logvar.weight[min_dim:, :],
                    mean=mean_weight, std=std_weight
                )
                
                mean_bias = global_model.decoder.fc_out.bias.mean().item()
                std_bias = global_model.decoder.fc_out.bias.std().item()
                nn.init.normal_(
                    customer_model.decoder.fc_out.bias[min_dim:],
                    mean=mean_bias, std=std_bias
                )
                nn.init.normal_(
                    customer_model.decoder.fc_logvar.bias[min_dim:],
                    mean=mean_bias, std=std_bias
                )
    
    # Create optimizer
    optimizer = optim.Adam(customer_model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, 
        threshold=1e-4, threshold_mode='rel'
    )
    
    # Create data loader
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_legit).to(device)),
        batch_size=min(batch_size, len(X_train_legit)),
        shuffle=True
    )
    
    # Training setup
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    
    # Training loop
    logger.info(f"Training for up to {epochs} epochs (early stopping patience: {early_stopping_patience})...")
    for epoch in range(epochs):
        customer_model.train()
        epoch_loss = 0.0
        
        # KL annealing factor
        kl_annealing_factor = min(1.0, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else 1.0
        
        # Beta warmup
        current_beta = min(beta, beta * epoch / beta_warmup_epochs) if beta_warmup_epochs > 0 else beta
        customer_model.beta = current_beta
        
        # Train one epoch
        for batch_idx, (data,) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass with current annealing and beta
            loss, recon_loss, kl_loss = customer_model.compute_loss(
                data, 
                annealing_factor=kl_annealing_factor
            )
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation on test data
        customer_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Process in batches to avoid OOM with large test sets
            for i in range(0, len(X_test_np), batch_size):
                end = min(i + batch_size, len(X_test_np))
                data = torch.from_numpy(X_test_np[i:end]).to(device)
                batch_loss, _, _ = customer_model.compute_loss(data)
                val_loss += batch_loss.item() * (end - i)
            
            val_loss /= len(X_test_np)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs} | " +
                      f"Train Loss: {epoch_loss:.4f} | " +
                      f"Val Loss: {val_loss:.4f} | " +
                      f"Beta: {current_beta:.2f} | " +
                      f"KL Anneal: {kl_annealing_factor:.2f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Create model directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the best model
            model_path = os.path.join(model_dir, f"customer_{customer_id}.pt")
            customer_model.save(model_path)
            logger.info(f"Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model for evaluation
    model_path = os.path.join(model_dir, f"customer_{customer_id}.pt")
    customer_model = CustomerVAE.load(model_path, input_dim=input_dim, customer_id=customer_id)
    customer_model.to(device)
    
    # Final model evaluation
    logger.info("Evaluating customer model on test set...")
    customer_model.eval()
    
    # Compute reconstruction errors on test set
    test_errors = []
    with torch.no_grad():
        for i in range(0, len(X_test_np), batch_size):
            end = min(i + batch_size, len(X_test_np))
            data = torch.from_numpy(X_test_np[i:end]).to(device)
            batch_errors = customer_model.compute_reconstruction_error(data)
            test_errors.extend(batch_errors.cpu().numpy())
    
    # Calculate optimal threshold
    threshold, f_beta_value = calculate_fbeta_threshold(
        y_test, np.array(test_errors), beta=fbeta
    )
    
    # Apply threshold
    y_pred = (np.array(test_errors) >= threshold).astype(int)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(y_test, y_pred, scores=test_errors)
    
    # Add model info to metrics
    metrics.update({
        "model_info": {
            "customer_id": int(customer_id),
            "transaction_count": len(customer_data),
            "fraud_count": int(customer_data['isFraud'].sum()) if has_fraud else 0,
            "input_dim": input_dim,
            "z_dim": z_dim,
            "hidden1": h1,
            "hidden2": h2,
            "beta": beta,
            "epochs_trained": best_epoch + 1,
            "best_val_loss": float(best_val_loss),
            "threshold": float(threshold),
            "fbeta_score": float(f_beta_value),
            "training_time_seconds": int(time.time() - start_time)
        }
    })
    
    # Save metrics
    metrics_dir = os.path.join(os.path.dirname(model_dir), "metrics", "customers")
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_path = os.path.join(metrics_dir, f"customer_{customer_id}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save ROC and PR curves
    curves_dir = os.path.join(metrics_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_errors)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(os.path.join(curves_dir, f"customer_{customer_id}_roc.csv"), index=False)
    
    # PR curve
    precision_values, recall_values, _ = precision_recall_curve(y_test, test_errors)
    pr_df = pd.DataFrame({"precision": precision_values, "recall": recall_values})
    pr_df.to_csv(os.path.join(curves_dir, f"customer_{customer_id}_pr.csv"), index=False)
    
    logger.info(f"Customer VAE for {customer_id} trained and evaluated.")
    logger.info(f"Results: Recall={metrics['recall']:.4f}, " + 
              f"Specificity={metrics['specificity']:.4f}, AUC={metrics['roc_auc']:.4f}")
    
    return customer_model, metrics

def train_all_customers(
    data_path: str,
    global_model_path: str,
    model_dir: str = "./models/customers",
    metrics_dir: str = "./artifacts/metrics",
    min_transactions: int = 100,
    max_customers: Optional[int] = None,
    customer_id_col: str = "card1",
    z_dim: int = 2,
    h1: int = 64,
    h2: int = 32,
    batch_size: int = 32,
    epochs: int = 50,
    beta: float = 1.0,
    fbeta: float = 30.0,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Train customer-specific VAE models with transfer learning from the global model.
    
    Args:
        data_path: Path to preprocessed data with customer features
        global_model_path: Path to pre-trained global VAE for transfer learning
        model_dir: Directory to save trained models
        metrics_dir: Directory to save metrics
        min_transactions: Minimum transactions required to train a customer model
        max_customers: Maximum number of customers to process (None for all)
        customer_id_col: Column name for customer IDs
        z_dim: Dimension of latent space
        h1: Size of first hidden layer
        h2: Size of second hidden layer
        batch_size: Training batch size
        epochs: Maximum number of training epochs
        beta: Beta value for KL term weighting
        fbeta: Beta value for F-beta score
        random_state: Random seed for reproducibility
        
    Returns:
        List of metrics dictionaries for each customer model
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("=== CUSTOMER MODEL TRAINING START ===")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading preprocessed data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded: {df.shape}")
    
    # Get input dimension from data
    data_cols = df.drop('isFraud', axis=1).select_dtypes(include=['number']).columns
    input_dim = len(data_cols)
    
    # Load global model for transfer learning
    logger.info(f"Loading global model from {global_model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        global_model = GlobalVAE.load(global_model_path, input_dim=input_dim)
        global_model.eval()  # Set to evaluation mode
        global_model.to(device)
    except Exception as e:
        logger.error(f"Error loading global model: {str(e)}")
        logger.error("Cannot proceed without global model for transfer learning. Exiting.")
        return []
    
    # Identify customers with sufficient data
    customer_counts = df[customer_id_col].value_counts()
    eligible_customers = customer_counts[customer_counts >= min_transactions].index.tolist()
    
    if max_customers is not None and max_customers < len(eligible_customers):
        logger.info(f"Limiting to {max_customers} customers (from {len(eligible_customers)} eligible)")
        eligible_customers = eligible_customers[:max_customers]
    else:
        logger.info(f"Training models for all {len(eligible_customers)} eligible customers")
    
    # Train models for each customer
    results = []
    
    for customer_id in tqdm(eligible_customers, desc="Training customer models"):
        logger.info(f"Processing customer {customer_id}")
        
        try:
            model, metrics = train_customer_vae(
                customer_id=customer_id,
                data=df,
                global_model=global_model,
                device=device,
                z_dim=z_dim,
                h1=h1,
                h2=h2,
                batch_size=batch_size,
                epochs=epochs,
                beta=beta,
                model_dir=model_dir,
                fbeta=fbeta,
                min_txns=min_transactions,
                random_state=random_state
            )
            
            if metrics:
                # Add customer ID to results
                metrics_summary = {
                    "customer_id": int(customer_id),
                    "transaction_count": metrics["model_info"]["transaction_count"],
                    "fraud_count": metrics["model_info"]["fraud_count"],
                    "threshold": metrics["model_info"]["threshold"],
                    "recall": metrics["recall"],
                    "specificity": metrics["specificity"],
                    "precision": metrics["precision"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics.get("pr_auc", 0.0)
                }
                
                results.append(metrics_summary)
        except Exception as e:
            logger.error(f"Error training model for customer {customer_id}: {str(e)}")
    
    # Save summary metrics for all customers
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(metrics_dir, "customer_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Calculate and log aggregate metrics
    if len(results) > 0:
        logger.info("=== CUSTOMER MODELS SUMMARY ===")
        logger.info(f"Total customers processed: {len(results)}")
        logger.info(f"Average Recall: {summary_df['recall'].mean():.4f}")
        logger.info(f"Average Specificity: {summary_df['specificity'].mean():.4f}")
        logger.info(f"Average ROC AUC: {summary_df['roc_auc'].mean():.4f}")
    else:
        logger.warning("No customer models were successfully trained.")
    
    logger.info("=== CUSTOMER MODEL TRAINING COMPLETE ===")
    
    return results

def main():
    """Main function for command-line execution."""
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train customer-specific VAE models")
    parser.add_argument("--data_dir", default="./data/processed", help="Directory with processed data")
    parser.add_argument("--input", default=None, help="Specific input file with customer features")
    parser.add_argument("--global_model", default="./models/global_vae.pt", help="Path to global model")
    parser.add_argument("--model_dir", default="./models/customers", help="Directory to save models")
    parser.add_argument("--metrics_dir", default="./artifacts/metrics", help="Directory to save metrics")
    parser.add_argument("--min_txns", type=int, default=100, help="Minimum transactions required")
    parser.add_argument("--max_customers", type=int, default=None, help="Maximum customers to process")
    parser.add_argument("--z_dim", type=int, default=2, help="Latent dimension size")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for KL term weighting")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Determine input path
    if args.input:
        data_path = args.input
    else:
        data_path = os.path.join(args.data_dir, "features_customer_train.csv")
    
    # Train customer models
    train_all_customers(
        data_path=data_path,
        global_model_path=args.global_model,
        model_dir=args.model_dir,
        metrics_dir=args.metrics_dir,
        min_transactions=args.min_txns,
        max_customers=args.max_customers,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        beta=args.beta,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()