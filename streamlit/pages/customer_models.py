"""
Customer models page: Analyze and compare customer-specific VAE models.

Shows:
- Performance distribution
- Customer model metrics
- Performance comparisons with global model
- Top performing customer models
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer VAE Models", 
    page_icon="👤",
    layout="wide"
)

# Page title
st.title("👤 Customer-Specific VAE Models")
st.markdown("""
This page presents the analysis of customer-specific VAE models trained with transfer learning from the global model.
Each model is tailored to the unique transaction patterns of individual customers, improving detection accuracy.
""")

# Function to load customer metrics
@st.cache_data
def load_customer_metrics(metrics_path="artifacts/metrics/customer_metrics_summary.csv"):
    """Load customer models metrics from CSV file."""
    try:
        if os.path.exists(metrics_path):
            return pd.read_csv(metrics_path)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading customer metrics: {str(e)}")
        return None

# Function to load global metrics for comparison
@st.cache_data
def load_global_metrics(metrics_path="artifacts/metrics/global_metrics.json"):
    """Load global model metrics for comparison."""
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading global metrics: {str(e)}")
        return None

# Load metrics
customer_metrics = load_customer_metrics()
global_metrics = load_global_metrics()

# Test metrics if available
test_customer_metrics = None
test_metrics_path = "artifacts/evaluation/test/customer_metrics.csv"
if os.path.exists(test_metrics_path):
    test_customer_metrics = pd.read_csv(test_metrics_path)

# Display customer model analysis
if customer_metrics is not None and not customer_metrics.empty:
    # Summary statistics
    st.header("Performance Overview")
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customer Models", f"{len(customer_metrics)}")
    with col2:
        st.metric("Avg. Recall", f"{customer_metrics['recall'].mean():.3f}")
    with col3:
        st.metric("Avg. Specificity", f"{customer_metrics['specificity'].mean():.3f}")
    with col4:
        st.metric("Avg. ROC AUC", f"{customer_metrics['roc_auc'].mean():.3f}")
    
    # Compare with global model if available
    if global_metrics:
        st.subheader("Comparison with Global Model")
        
        # Create comparison dataframe
        comparison_data = {
            'Metric': ['Recall', 'Specificity', 'ROC AUC', 'Precision'],
            'Global Model': [
                global_metrics['performance']['recall'],
                global_metrics['performance']['specificity'],
                global_metrics['performance']['roc_auc'],
                global_metrics['performance']['precision']
            ],
            'Customer Models (Mean)': [
                customer_metrics['recall'].mean(),
                customer_metrics['specificity'].mean(),
                customer_metrics['roc_auc'].mean(),
                customer_metrics['precision'].mean()
            ]
        }
        
        # Add test metrics if available
        if test_customer_metrics is not None:
            comparison_data['Customer Models (Test)'] = [
                test_customer_metrics['recall'].mean(),
                test_customer_metrics['specificity'].mean(),
                test_customer_metrics['roc_auc'].mean(),
                test_customer_metrics['precision'].mean()
            ]
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Recall comparison
        axes[0].bar(['Global Model', 'Customer Models'], 
                 [global_metrics['performance']['recall'], customer_metrics['recall'].mean()],
                 color=['blue', 'green'])
        axes[0].set_title('Recall Comparison')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Specificity comparison
        axes[1].bar(['Global Model', 'Customer Models'], 
                 [global_metrics['performance']['specificity'], customer_metrics['specificity'].mean()],
                 color=['blue', 'green'])
        axes[1].set_title('Specificity Comparison')
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Performance distribution
    st.header("Performance Distribution")
    
    # Allow selecting which metric to view
    metrics_to_plot = st.selectbox(
        "Select metric to visualize:", 
        ['recall', 'specificity', 'roc_auc', 'precision', 'f1_score'],
        index=0
    )
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(customer_metrics[metrics_to_plot], kde=True, ax=ax)
    
    # Add global model line if available
    if global_metrics and metrics_to_plot in ['recall', 'specificity', 'roc_auc', 'precision']:
        if metrics_to_plot == 'recall':
            global_value = global_metrics['performance']['recall']
        elif metrics_to_plot == 'specificity':
            global_value = global_metrics['performance']['specificity']
        elif metrics_to_plot == 'roc_auc':
            global_value = global_metrics['performance']['roc_auc']
        elif metrics_to_plot == 'precision':
            global_value = global_metrics['performance']['precision']
        
        ax.axvline(x=global_value, color='red', linestyle='--', 
                  label=f'Global Model: {global_value:.3f}')
        ax.legend()
    
    ax.set_title(f'Distribution of {metrics_to_plot.capitalize()} Across Customer Models')
    ax.set_xlabel(metrics_to_plot.capitalize())
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Top performing models
    st.header("Top Performing Customer Models")
    
    # Select metric for ranking
    ranking_metric = st.selectbox(
        "Rank by:", 
        ['recall', 'specificity', 'roc_auc', 'f1_score'],
        index=0
    )
    
    # Get top models
    top_models = customer_metrics.sort_values(ranking_metric, ascending=False).head(10)
    
    # Display table
    st.dataframe(
        top_models[['customer_id', 'transaction_count', 'fraud_count', 
                  'recall', 'specificity', 'roc_auc', 'precision']],
        use_container_width=True
    )
    
    # Performance vs. transaction count
    st.header("Performance vs. Transaction Count")
    
    # Select metrics for scatter plot
    scatter_metric = st.selectbox(
        "Performance metric:", 
        ['recall', 'specificity', 'roc_auc', 'precision'],
        index=0
    )
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        customer_metrics['transaction_count'], 
        customer_metrics[scatter_metric],
        alpha=0.6,
        c=customer_metrics['fraud_count'],
        cmap='viridis'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Fraud Count')
    
    # Add global model line if available
    if global_metrics and scatter_metric in ['recall', 'specificity', 'roc_auc', 'precision']:
        if scatter_metric == 'recall':
            global_value = global_metrics['performance']['recall']
        elif scatter_metric == 'specificity':
            global_value = global_metrics['performance']['specificity']
        elif scatter_metric == 'roc_auc':
            global_value = global_metrics['performance']['roc_auc']
        elif scatter_metric == 'precision':
            global_value = global_metrics['performance']['precision']
        
        ax.axhline(y=global_value, color='red', linestyle='--', 
                  label=f'Global Model: {global_value:.3f}')
        ax.legend()
    
    ax.set_title(f'{scatter_metric.capitalize()} vs. Transaction Count')
    ax.set_xlabel('Transaction Count')
    ax.set_ylabel(scatter_metric.capitalize())
    ax.grid(True, alpha=0.3)
    
    # Use log scale for transaction count if range is large
    if customer_metrics['transaction_count'].max() / customer_metrics['transaction_count'].min() > 10:
        ax.set_xscale('log')
        ax.set_xlabel('Transaction Count (log scale)')
    
    st.pyplot(fig)
    
    # Detailed customer model selection
    st.header("Individual Customer Model Details")
    
    # Select customer ID
    selected_customer = st.selectbox(
        "Select customer ID:",
        customer_metrics['customer_id'].tolist(),
        index=0
    )
    
    # Get metrics for selected customer
    customer_row = customer_metrics[customer_metrics['customer_id'] == selected_customer].iloc[0]
    
    # Display customer-specific metrics
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        st.subheader(f"Customer {selected_customer} Metrics")
        
        # Create metrics dataframe
        customer_detail_df = pd.DataFrame({
            'Metric': ['Transactions', 'Fraud Count', 'Recall', 'Specificity', 
                     'ROC AUC', 'Precision', 'F1 Score', 'Threshold'],
            'Value': [
                customer_row['transaction_count'],
                customer_row['fraud_count'],
                f"{customer_row['recall']:.4f}",
                f"{customer_row['specificity']:.4f}",
                f"{customer_row['roc_auc']:.4f}",
                f"{customer_row['precision']:.4f}",
                f"{customer_row['f1_score']:.4f}",
                f"{customer_row['threshold']:.4f}"
            ]
        })
        
        st.dataframe(customer_detail_df, use_container_width=True)
    
    # Check if customer curves are available
    with detail_col2:
        customer_roc_path = f"artifacts/metrics/curves/customer_{selected_customer}_roc.csv"
        customer_pr_path = f"artifacts/metrics/curves/customer_{selected_customer}_pr.csv"
        
        if os.path.exists(customer_roc_path) and os.path.exists(customer_pr_path):
            customer_roc = pd.read_csv(customer_roc_path)
            customer_pr = pd.read_csv(customer_pr_path)
            
            # Create tabs for ROC and PR curves
            roc_tab, pr_tab = st.tabs(["ROC Curve", "Precision-Recall Curve"])
            
            with roc_tab:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(customer_roc['fpr'], customer_roc['tpr'], 
                      label=f'Customer {selected_customer} (AUC={customer_row["roc_auc"]:.3f})')
                
                # Add global ROC if available
                if global_metrics and os.path.exists("artifacts/metrics/global_roc.csv"):
                    global_roc = pd.read_csv("artifacts/metrics/global_roc.csv")
                    ax.plot(global_roc['fpr'], global_roc['tpr'], 
                          label=f'Global Model (AUC={global_metrics["performance"]["roc_auc"]:.3f})',
                          linestyle='--')
                
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax.set_title('ROC Curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            
            with pr_tab:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(customer_pr['recall'], customer_pr['precision'], 
                      label=f'Customer {selected_customer}')
                
                # Add global PR if available
                if global_metrics and os.path.exists("artifacts/metrics/global_pr.csv"):
                    global_pr = pd.read_csv("artifacts/metrics/global_pr.csv")
                    ax.plot(global_pr['recall'], global_pr['precision'], 
                          label='Global Model',
                          linestyle='--')
                
                ax.set_title('Precision-Recall Curve')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
        else:
            st.info("Detailed curves for this customer model are not available.")
else:
    st.warning("Customer model metrics not found. Run the training pipeline first to generate metrics.")
    
    # Show instructions
    st.markdown("""
    ### Getting Started
    
    To train customer-specific models and generate metrics:
    
    ```bash
    # Run the training pipeline
    python -m doit train_customer_models
    
    # Or run the entire pipeline
    python -m doit
    ```
    
    After training, return to this page to view the customer model performance and analysis.
    """)