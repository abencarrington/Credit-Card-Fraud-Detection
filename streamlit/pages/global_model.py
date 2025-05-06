"""
Global model page: Visualize and analyze the global VAE model performance.

Shows:
- Model metrics
- Performance curves
- Feature importance
- Reconstruction error distribution
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.models.vae import GlobalVAE

# Page configuration
st.set_page_config(
    page_title="Global VAE Model", 
    page_icon="🌍",
    layout="wide"
)

# Page title
st.title("🌍 Global VAE Model Analysis")
st.markdown("""
This page presents the performance and analysis of the global VAE model trained on all legitimate transactions.
The model serves as the foundation for identifying anomalous transactions based on reconstruction error.
""")

# Function to load metrics
@st.cache_data
def load_metrics(metrics_path="artifacts/metrics/global_metrics.json"):
    """Load model metrics from JSON file."""
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        return None

# Function to load curves
@st.cache_data
def load_curves(curve_type="roc", path="artifacts/metrics"):
    """Load ROC or PR curve data."""
    try:
        curve_path = os.path.join(path, f"global_{curve_type}.csv")
        return pd.read_csv(curve_path)
    except Exception as e:
        st.error(f"Error loading {curve_type} curve: {str(e)}")
        return None

# Load model and metrics
metrics = load_metrics()
roc_curve = load_curves("roc")
pr_curve = load_curves("pr")

# Evaluation on test set
test_metrics = None
test_roc = None
test_pr = None
test_path = "artifacts/evaluation/test/global_metrics.json"
if os.path.exists(test_path):
    with open(test_path, "r") as f:
        test_metrics = json.load(f)
    test_roc_path = "artifacts/evaluation/test/curves/global_roc.csv"
    test_pr_path = "artifacts/evaluation/test/curves/global_pr.csv"
    if os.path.exists(test_roc_path) and os.path.exists(test_pr_path):
        test_roc = pd.read_csv(test_roc_path)
        test_pr = pd.read_csv(test_pr_path)

# Display model information
if metrics:
    # Main metrics cards
    st.header("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fraud Recall", f"{metrics['performance']['recall']:.3f}")
    with col2:
        st.metric("Specificity", f"{metrics['performance']['specificity']:.3f}")
    with col3:
        st.metric("ROC AUC", f"{metrics['performance']['roc_auc']:.3f}")
    with col4:
        st.metric("F-Beta Score (β=30)", f"{metrics['performance']['f_beta']:.3f}")
    
    # If test metrics available, show comparison
    if test_metrics:
        st.subheader("Train vs Test Performance")
        df_comparison = pd.DataFrame({
            'Metric': ['Recall', 'Specificity', 'ROC AUC', 'Precision'],
            'Train': [
                metrics['performance']['recall'], 
                metrics['performance']['specificity'],
                metrics['performance']['roc_auc'],
                metrics['performance']['precision']
            ],
            'Test': [
                test_metrics['recall'],
                test_metrics['specificity'],
                test_metrics['roc_auc'],
                test_metrics['precision']
            ]
        })
        st.dataframe(df_comparison, use_container_width=True)
    
    # Model architecture details
    st.header("Model Architecture")
    
    # Extract model parameters
    model_params = metrics.get('model_params', {})
    
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("### Architecture Parameters")
        params_df = pd.DataFrame({
            'Parameter': ['Input Dimension', 'Hidden Layer 1', 'Hidden Layer 2', 'Latent Dimension', 'Beta Value'],
            'Value': [
                model_params.get('input_dim', 'N/A'),
                model_params.get('hidden1', 'N/A'),
                model_params.get('hidden2', 'N/A'),
                model_params.get('z_dim', 'N/A'),
                model_params.get('beta', 'N/A')
            ]
        })
        st.dataframe(params_df, use_container_width=True)
    
    with arch_col2:
        st.markdown("### Training Details")
        training_df = pd.DataFrame({
            'Parameter': ['Epochs Trained', 'Final Learning Rate', 'Best Validation Loss', 'Training Time (seconds)'],
            'Value': [
                model_params.get('epochs_trained', 'N/A'),
                model_params.get('final_lr', 'N/A'),
                metrics['performance'].get('best_val_loss', 'N/A'),
                metrics.get('training_time_seconds', 'N/A')
            ]
        })
        st.dataframe(training_df, use_container_width=True)
    
    # Performance curves
    st.header("Performance Curves")
    
    curves_col1, curves_col2 = st.columns(2)
    
    with curves_col1:
        if roc_curve is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(roc_curve['fpr'], roc_curve['tpr'], label='Train')
            
            # Add test curve if available
            if test_roc is not None:
                ax.plot(test_roc['fpr'], test_roc['tpr'], label='Test')
                ax.legend()
                
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_title('ROC Curve')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with curves_col2:
        if pr_curve is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(pr_curve['recall'], pr_curve['precision'], label='Train')
            
            # Add test curve if available
            if test_pr is not None:
                ax.plot(test_pr['recall'], test_pr['precision'], label='Test')
                ax.legend()
                
            ax.set_title('Precision-Recall Curve')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # Reconstruction error distribution
    st.header("Reconstruction Error Distribution")
    
    # Check if error distribution image exists
    error_dist_path = "streamlit/assets/error_distribution.png"
    if os.path.exists(error_dist_path):
        st.image(error_dist_path, caption="Distribution of Reconstruction Errors")
    else:
        error_dist_path = "artifacts/evaluation/test/curves/global_error_dist.png"
        if os.path.exists(error_dist_path):
            st.image(error_dist_path, caption="Test Data: Distribution of Reconstruction Errors")
        else:
            st.info("Error distribution visualization not available. Run the full evaluation to generate this visualization.")
    
    # Feature importance
    st.header("Feature Importance")
    
    # Check if SHAP visualization exists
    shap_path = "artifacts/metrics/global_shap.png"
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Feature Importance")
    else:
        shap_path = "artifacts/evaluation/test/global_shap.png"
        if os.path.exists(shap_path):
            st.image(shap_path, caption="Test Data: SHAP Feature Importance")
        else:
            st.info("SHAP feature importance visualization not available. Run the full evaluation to generate this visualization.")
    
    # Feature importance from CSV
    fi_path = "artifacts/evaluation/test/global_feature_importance.csv"
    if os.path.exists(fi_path):
        st.subheader("Top Features by Importance")
        fi_df = pd.read_csv(fi_path)
        fi_df = fi_df.head(20)  # Show top 20
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=fi_df)
        ax.set_title('Top 20 Features by Importance')
        ax.set_xlabel('Importance')
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
    
    # Confusion matrix if available
    if 'confusion_matrix' in metrics:
        st.header("Confusion Matrix")
        
        cm = metrics['confusion_matrix']
        cm_df = pd.DataFrame([
            ['True Negative', 'False Positive'],
            ['False Negative', 'True Positive']
        ])
        
        # Format with values
        cm_vals = [
            [cm['true_negatives'], cm['false_positives']],
            [cm['false_negatives'], cm['true_positives']]
        ]
        
        # Create styled dataframe
        def color_cm(val):
            color = ''
            if isinstance(val, str):
                if 'True' in val:
                    color = 'background-color: rgba(0, 255, 0, 0.2)'
                elif 'False' in val:
                    color = 'background-color: rgba(255, 0, 0, 0.2)'
            return color
        
        st.dataframe(
            cm_df.style.applymap(color_cm),
            use_container_width=True
        )
        
        # Display values
        st.markdown(f"""
        - **True Negatives**: {cm['true_negatives']} (correctly identified legitimate transactions)
        - **False Positives**: {cm['false_positives']} (legitimate transactions incorrectly flagged as fraud)
        - **False Negatives**: {cm['false_negatives']} (fraudulent transactions missed)
        - **True Positives**: {cm['true_positives']} (correctly identified fraudulent transactions)
        """)
    
    # Latent space visualization (if z_dim = 2)
    latent_path = "artifacts/evaluation/test/global_latent_space.png"
    if os.path.exists(latent_path) and model_params.get('z_dim', 0) == 2:
        st.header("Latent Space Visualization")
        st.image(latent_path, caption="2D Latent Space Representation")
        st.markdown("""
        This plot shows the 2D latent space representation of transactions, color-coded by fraud status.
        Clustering patterns in this space indicate how well the model has learned to separate fraudulent from legitimate transactions.
        """)
else:
    st.warning("Model metrics not found. Run the training pipeline first to generate metrics.")
    
    # Show instructions
    st.markdown("""
    ### Getting Started
    
    To train the global model and generate metrics:
    
    ```bash
    # Run the training pipeline
    python -m doit train_global_model
    
    # Or run the entire pipeline
    python -m doit
    ```
    
    After training, return to this page to view the model performance and analysis.
    """)