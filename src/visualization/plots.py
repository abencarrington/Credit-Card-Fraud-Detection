"""
Enhanced visualization module with plotting utilities for:
1. Training curves
2. ROC and PR curves
3. Reconstruction error distributions
4. Latent space visualization
5. Feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Union
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.manifold import TSNE
import umap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


def plot_training_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    kl_losses: Optional[List[float]] = None,
    recon_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        kl_losses: Optional list of KL divergence losses
        recon_losses: Optional list of reconstruction losses
        save_path: Path to save the figure (if None, just display)
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    if kl_losses is not None and recon_losses is not None:
        # Create secondary y-axis for component losses
        ax2 = ax1.twinx()
        ax2.plot(epochs, kl_losses, 'g--', label='KL Loss')
        ax2.plot(epochs, recon_losses, 'c--', label='Recon Loss')
        ax2.set_ylabel('Component Loss')
        ax2.tick_params(axis='y')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training and Validation Loss Curves')
    ax1.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_reconstruction_error_distribution(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    percentiles: List[float] = [90, 95, 97.5, 99],
    save_path: Optional[str] = None
) -> None:
    """
    Plot histogram of reconstruction errors with threshold.
    
    Args:
        errors: Array of reconstruction errors
        labels: Array of true labels (0 for normal, 1 for fraud)
        threshold: Classification threshold
        percentiles: Percentiles to mark on the plot
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Get errors for each class
    normal_errors = errors[labels == 0]
    fraud_errors = errors[labels == 1]
    
    # Compute percentile thresholds
    percentile_thresholds = [np.percentile(normal_errors, p) for p in percentiles]
    
    # Plot histograms
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal', color='green', density=True)
    plt.hist(fraud_errors, bins=50, alpha=0.5, label='Fraud', color='red', density=True)
    
    # Plot optimal threshold
    plt.axvline(x=threshold, color='blue', linestyle='-', linewidth=2, 
                label=f'Optimal Threshold: {threshold:.2f}')
    
    # Plot percentile thresholds
    for p, t in zip(percentiles, percentile_thresholds):
        plt.axvline(x=t, color='gray', linestyle='--', alpha=0.7,
                    label=f'{p}th Percentile: {t:.2f}')
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pr_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    beta: float = 1.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot precision-recall and ROC curves side by side.
    
    Args:
        y_true: Array of true labels (0 for normal, 1 for fraud)
        scores: Array of anomaly scores
        threshold: Classification threshold
        beta: F-beta parameter
        save_path: Path to save the figure
    """
    # Compute PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find threshold points on curves
    if len(pr_thresholds) > 0:
        # For PR curve, find closest threshold
        idx_pr = np.abs(pr_thresholds - threshold).argmin()
        if idx_pr < len(precision) - 1:  # Ensure index is valid
            pr_point = (recall[idx_pr], precision[idx_pr])
        else:
            pr_point = (recall[-1], precision[-1])
    else:
        pr_point = (0, 0)
        
    if len(roc_thresholds) > 0:
        # For ROC curve, find closest threshold
        idx_roc = np.abs(roc_thresholds - threshold).argmin()
        roc_point = (fpr[idx_roc], tpr[idx_roc])
    else:
        roc_point = (0, 0)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PR curve
    ax1.plot(recall, precision, 'b-', label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax1.scatter(pr_point[0], pr_point[1], color='red', s=100, marker='o',
               label=f'Threshold = {threshold:.2f}')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'Precision-Recall Curve (β = {beta})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ROC curve
    ax2.plot(fpr, tpr, 'g-', label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax2.scatter(roc_point[0], roc_point[1], color='red', s=100, marker='o',
               label=f'Threshold = {threshold:.2f}')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_latent_space(
    latent_points: np.ndarray,
    labels: np.ndarray,
    method: str = 'raw',
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the latent space representation.
    
    Args:
        latent_points: Array of latent space coordinates (n_samples, z_dim)
        labels: Array of true labels (0 for normal, 1 for fraud)
        method: Visualization method ('raw', 'tsne', or 'umap')
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # If latent space has more than 2 dimensions, use dimensionality reduction
    if latent_points.shape[1] > 2:
        if method == 'tsne':
            # t-SNE projection
            reducer = TSNE(n_components=2, random_state=42)
            points_2d = reducer.fit_transform(latent_points)
            title = 't-SNE Projection of Latent Space'
        elif method == 'umap':
            # UMAP projection
            reducer = umap.UMAP(n_components=2, random_state=42)
            points_2d = reducer.fit_transform(latent_points)
            title = 'UMAP Projection of Latent Space'
        else:
            # Use first two dimensions
            points_2d = latent_points[:, :2]
            title = 'First Two Dimensions of Latent Space'
    else:
        # Already 2D
        points_2d = latent_points
        title = '2D Latent Space Representation'
    
    # Separate points by class
    normal_points = points_2d[labels == 0]
    fraud_points = points_2d[labels == 1]
    
    # Plot points
    plt.scatter(normal_points[:, 0], normal_points[:, 1], alpha=0.6, s=5,
                label='Normal', color='green')
    plt.scatter(fraud_points[:, 0], fraud_points[:, 1], alpha=0.8, s=15,
                label='Fraud', color='red')
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    title: str = 'Feature Importance',
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance values.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        title: Plot title
        top_n: Number of top features to display
        save_path: Path to save the figure
    """
    # Create DataFrame for sorting
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    })
    
    # Sort and get top features
    df = df.sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=df, palette='viridis')
    
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_customer_performance_distribution(
    metrics_df: pd.DataFrame,
    metric_name: str,
    bins: int = 20,
    global_value: Optional[float] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of performance metrics across customers.
    
    Args:
        metrics_df: DataFrame with customer metrics
        metric_name: Name of the metric to plot
        bins: Number of histogram bins
        global_value: Value of the metric for the global model (optional)
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(metrics_df[metric_name], bins=bins, kde=True)
    
    if global_value is not None:
        plt.axvline(x=global_value, color='red', linestyle='--', linewidth=2,
                   label=f'Global Model: {global_value:.3f}')
        plt.legend()
    
    plt.title(f'Distribution of {metric_name.capitalize()} Across Customers')
    plt.xlabel(metric_name.capitalize())
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_reconstruction_examples(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: List[str],
    n_examples: int = 3,
    save_path: Optional[str] = None
) -> None:
    """
    Plot original vs reconstructed examples side by side.
    
    Args:
        original: Original data points
        reconstructed: Reconstructed data points
        feature_names: Names of features
        n_examples: Number of examples to plot
        save_path: Path to save the figure
    """
    n_examples = min(n_examples, original.shape[0])
    n_features = min(20, original.shape[1])  # Limit to 20 features for readability
    
    # Select top features by variance
    if original.shape[1] > n_features:
        var = np.var(original, axis=0)
        top_idx = np.argsort(-var)[:n_features]
        orig_subset = original[:, top_idx]
        recon_subset = reconstructed[:, top_idx]
        feature_subset = [feature_names[i] for i in top_idx]
    else:
        orig_subset = original
        recon_subset = reconstructed
        feature_subset = feature_names
    
    # Create plot
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 4 * n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i in range(n_examples):
        df = pd.DataFrame({
            'Feature': feature_subset * 2,
            'Value': np.concatenate([orig_subset[i], recon_subset[i]]),
            'Type': ['Original'] * n_features + ['Reconstructed'] * n_features
        })
        
        sns.barplot(x='Feature', y='Value', hue='Type', data=df, ax=axes[i])
        axes[i].set_title(f'Example {i+1}: Original vs Reconstructed')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        axes[i].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()