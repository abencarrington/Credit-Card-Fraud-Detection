"""
Metrics module for fraud detection evaluation.

Includes functions for:
1. F-beta threshold calculation
2. Performance metrics computation
3. Threshold optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Any, Union
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    recall_score, precision_score, f1_score, fbeta_score,
    confusion_matrix, classification_report
)


def calculate_fbeta_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    beta: float = 30.0,
    min_recall: float = 0.90
) -> Tuple[float, float]:
    """
    Calculate optimal threshold based on F-beta score.
    
    Heavily weighs recall over precision for fraud detection scenarios.
    Ensures minimum recall level is met.
    
    Args:
        y_true: True labels (0=normal, 1=fraud)
        scores: Anomaly scores (higher is more anomalous)
        beta: Beta parameter for F-beta (higher values favor recall over precision)
        min_recall: Minimum acceptable recall
        
    Returns:
        threshold: Optimal threshold value
        fbeta_value: F-beta score at the optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Calculate F-beta score for each threshold
    # Note: precision_recall_curve returns one more precision/recall value than thresholds
    # So we need to match the array sizes
    fbeta = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        y_pred = (scores >= t).astype(int)
        fbeta[i] = fbeta_score(y_true, y_pred, beta=beta)
    
    # Filter thresholds by minimum recall requirement
    valid_indices = []
    for i, t in enumerate(thresholds):
        y_pred = (scores >= t).astype(int)
        r = recall_score(y_true, y_pred)
        if r >= min_recall:
            valid_indices.append(i)
    
    if not valid_indices:
        # Fall back to minimum threshold if no valid thresholds found
        return thresholds.min(), 0.0
    
    # Get threshold with highest F-beta score among valid thresholds
    valid_fbeta = fbeta[valid_indices]
    best_idx = valid_indices[valid_fbeta.argmax()]
    
    return thresholds[best_idx], fbeta[best_idx]


def calculate_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True labels (0=normal, 1=fraud)
        y_pred: Predicted labels (0=normal, 1=fraud)
        scores: Raw anomaly scores for AUC calculation (optional)
        
    Returns:
        Dictionary of performance metrics
    """
    # Basic classification metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    # Add AUC if scores are provided
    metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "fbeta_score_10": fbeta_score(y_true, y_pred, beta=10.0),
        "fbeta_score_30": fbeta_score(y_true, y_pred, beta=30.0),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "positive_predictive_value": precision,
        "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "false_discovery_rate": fp / (fp + tp) if (fp + tp) > 0 else 0,
        "false_omission_rate": fn / (tn + fn) if (tn + fn) > 0 else 0
    }
    
    if scores is not None:
        # AUC metrics
        fpr, tpr, _ = roc_curve(y_true, scores)
        metrics["roc_auc"] = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
        metrics["pr_auc"] = auc(recall_curve, precision_curve)
    
    return metrics


def evaluate_at_multiple_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_percentiles: List[float] = [90, 95, 97.5, 99],
    custom_thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Evaluate model performance at multiple thresholds.
    
    Args:
        y_true: True labels (0=normal, 1=fraud)
        scores: Anomaly scores (higher is more anomalous)
        threshold_percentiles: Percentiles of normal scores to use as thresholds
        custom_thresholds: Custom threshold values to evaluate
        
    Returns:
        DataFrame with metrics at each threshold
    """
    results = []
    
    # Get normal class scores for percentile thresholds
    normal_scores = scores[y_true == 0]
    
    # Add percentile-based thresholds
    for percentile in threshold_percentiles:
        threshold = np.percentile(normal_scores, percentile)
        y_pred = (scores >= threshold).astype(int)
        
        metrics = calculate_performance_metrics(y_true, y_pred, scores)
        metrics["threshold"] = threshold
        metrics["threshold_type"] = f"{percentile}th Percentile"
        
        results.append(metrics)
    
    # Add custom thresholds if provided
    if custom_thresholds:
        for threshold in custom_thresholds:
            y_pred = (scores >= threshold).astype(int)
            
            metrics = calculate_performance_metrics(y_true, y_pred, scores)
            metrics["threshold"] = threshold
            metrics["threshold_type"] = "Custom"
            
            results.append(metrics)
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def threshold_sensitivity_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 100,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Analyze sensitivity of metrics to threshold value.
    
    Args:
        y_true: True labels (0=normal, 1=fraud)
        scores: Anomaly scores (higher is more anomalous)
        n_thresholds: Number of thresholds to evaluate
        save_path: Path to save the sensitivity plot
        
    Returns:
        thresholds: Array of evaluated thresholds
        metrics: Dictionary of metric arrays corresponding to thresholds
    """
    # Generate evenly spaced thresholds between min and max score
    min_score, max_score = scores.min(), scores.max()
    thresholds = np.linspace(min_score, max_score, n_thresholds)
    
    # Initialize metric arrays
    recalls = np.zeros(n_thresholds)
    precisions = np.zeros(n_thresholds)
    specificities = np.zeros(n_thresholds)
    f1_scores = np.zeros(n_thresholds)
    fbeta_10 = np.zeros(n_thresholds)
    fbeta_30 = np.zeros(n_thresholds)
    
    # Calculate metrics for each threshold
    for i, threshold in enumerate(thresholds):
        y_pred = (scores >= threshold).astype(int)
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        recalls[i] = recall_score(y_true, y_pred) if tp + fn > 0 else 0
        precisions[i] = precision_score(y_true, y_pred) if tp + fp > 0 else 0
        specificities[i] = tn / (tn + fp) if tn + fp > 0 else 0
        f1_scores[i] = f1_score(y_true, y_pred)
        fbeta_10[i] = fbeta_score(y_true, y_pred, beta=10.0)
        fbeta_30[i] = fbeta_score(y_true, y_pred, beta=30.0)
    
    # Package results
    metrics = {
        'recall': recalls,
        'precision': precisions,
        'specificity': specificities,
        'f1_score': f1_scores,
        'fbeta_10': fbeta_10,
        'fbeta_30': fbeta_30
    }
    
    # Optional plot
    if save_path:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, recalls, 'r-', label='Recall')
        plt.plot(thresholds, specificities, 'b-', label='Specificity')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Recall & Specificity vs Threshold')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, precisions, 'g-', label='Precision')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision vs Threshold')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(thresholds, f1_scores, 'k-', label='F1')
        plt.plot(thresholds, fbeta_10, 'm-', label='F10')
        plt.plot(thresholds, fbeta_30, 'c-', label='F30')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('F-Scores vs Threshold')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(recalls, precisions, 'b-')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Trade-off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return thresholds, metrics


def compare_models(
    model_results: Dict[str, Dict[str, float]],
    metrics_to_compare: List[str] = ['recall', 'precision', 'specificity', 'roc_auc'],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models across key metrics.
    
    Args:
        model_results: Dictionary of model names to metrics dictionaries
        metrics_to_compare: List of metrics to include in comparison
        save_path: Path to save the comparison plot
        
    Returns:
        DataFrame with model comparison
    """
    # Build comparison DataFrame
    data = []
    for model_name, metrics in model_results.items():
        row = {'model': model_name}
        for metric in metrics_to_compare:
            if metric in metrics:
                row[metric] = metrics[metric]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Optional visualization
    if save_path:
        metrics_count = len(metrics_to_compare)
        fig_width = max(10, metrics_count * 2)
        
        plt.figure(figsize=(fig_width, 6))
        
        # Bar plot for each metric
        for i, metric in enumerate(metrics_to_compare):
            if metric in df.columns:
                plt.subplot(1, metrics_count, i+1)
                sns_plot = sns.barplot(x='model', y=metric, data=df)
                for item in sns_plot.get_xticklabels():
                    item.set_rotation(45)
                plt.title(f'{metric.capitalize()}')
                plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return df


def optimal_threshold_search(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_recall: float = 0.95,
    fbeta: float = 30.0
) -> Dict[str, Any]:
    """
    Find optimal threshold with specified constraints.
    
    Args:
        y_true: True labels (0=normal, 1=fraud)
        scores: Anomaly scores
        min_recall: Minimum acceptable recall
        fbeta: Beta for F-beta score
        
    Returns:
        Dictionary with threshold and metrics
    """
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Find thresholds that satisfy minimum recall
    valid_indices = [i for i, r in enumerate(recall[:-1]) if r >= min_recall]
    
    if not valid_indices:
        # If no threshold satisfies minimum recall, return default
        return {
            'threshold': scores.min(),
            'satisfies_constraints': False,
            'recall': recall[-1],
            'precision': precision[-1],
            'fbeta': 0.0
        }
    
    # Calculate F-beta for valid thresholds
    fbeta_scores = []
    for i in valid_indices:
        pred = (scores >= thresholds[i]).astype(int)
        fbeta_scores.append(fbeta_score(y_true, pred, beta=fbeta))
    
    # Find best threshold
    best_idx = valid_indices[np.argmax(fbeta_scores)]
    optimal_threshold = thresholds[best_idx]
    
    # Calculate final metrics with optimal threshold
    y_pred = (scores >= optimal_threshold).astype(int)
    final_metrics = calculate_performance_metrics(y_true, y_pred, scores)
    
    return {
        'threshold': optimal_threshold,
        'satisfies_constraints': True,
        'metrics': final_metrics,
        'fbeta': fbeta_scores[np.argmax(fbeta_scores)]
    }