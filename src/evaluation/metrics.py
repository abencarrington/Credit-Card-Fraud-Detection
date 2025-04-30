"""
Plotting utilities for Precision-Recall and ROC curves,
and F-Beta threshold selection.
"""

import logging
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

logger = logging.getLogger(__name__)

def plot_pr_roc(y_true, scores, beta=30.0):
    """
    Compute PR & ROC curves, find threshold maximizing F-Beta,
    and display both plots side by side.

    Args:
        y_true: 1D array of true labels.
        scores: 1D array of anomaly scores.
        beta:   F-Beta beta parameter.

    Returns:
        Optimal threshold for anomaly detection.
    """
    prec, rec, thr_pr = precision_recall_curve(y_true, scores)
    fbeta = (1+beta**2)*prec*rec/(beta**2*prec + rec + 1e-8)
    idx = fbeta.argmax()
    t_opt = thr_pr[idx]

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr,tpr)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    ax1.plot(rec,prec, label="PR")
    ax1.scatter(rec[idx],prec[idx],c="red", label=f"β-thr={t_opt:.2e}")
    ax1.set_title("Precision-Recall Curve")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend(); ax1.grid()

    ax2.plot(fpr,tpr, label=f"ROC (AUC={roc_auc:.3f})")
    ax2.plot([0,1],[0,1],"--",color="gray")
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.legend(); ax2.grid()

    plt.tight_layout(); plt.show()
    return t_opt