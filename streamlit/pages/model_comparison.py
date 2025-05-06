"""
Enhanced comparison page for the IEEE-CIS fraud detection dashboard.

Features:
- Side-by-side metrics & curves comparison between global and customer models
- Performance distribution visualization
- Feature importance comparison
- Test vs. Train performance comparison
- Interactive customer selection
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache helper functions
@st.cache_data
def load_global_metrics(metrics_path="artifacts/metrics/global_metrics.json"):
    """Load global model metrics from JSON file."""
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading global metrics: {str(e)}")
        st.error(f"Error loading global metrics: {str(e)}")
        return None

@st.cache_data
def load_customer_metrics(metrics_path="artifacts/metrics/customer_metrics_summary.csv"):
    """Load customer metrics summary from CSV file."""
    try:
        return pd.read_csv(metrics_path)
    except Exception as e:
        logger.error(f"Error loading customer metrics: {str(e)}")
        st.error(f"Error loading customer metrics: {str(e)}")
        return None

@st.cache_data
def load_test_metrics(metrics_path="artifacts/evaluation/test/global_metrics.json"):
    """Load test evaluation metrics."""
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading test metrics: {str(e)}")
        return None

@st.cache_data
def load_curves(curve_type="roc", model_type="global", customer_id=None):
    """
    Load ROC or PR curve data.
    
    Args:
        curve_type: 'roc' or 'pr'
        model_type: 'global' or 'customer'
        customer_id: Required if model_type is 'customer'
    
    Returns:
        DataFrame with curve data
    """
    try:
        if model_type == "global":
            path = f"artifacts/metrics/global_{curve_type}.csv"
        else:
            path = f"artifacts/metrics/customers/customer_{customer_id}_{curve_type}.csv"
            
        if not os.path.exists(path):
            logger.warning(f"Curve file not found: {path}")
            return None
            
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading {curve_type} curve: {str(e)}")
        return None

@st.cache_data
def load_feature_importance(model_type="global", customer_id=None):
    """Load feature importance data."""
    try:
        if model_type == "global":
            path = f"artifacts/metrics/global_feature_importance.csv"
        else:
            path = f"artifacts/metrics/customers/customer_{customer_id}_feature_importance.csv"
            
        if not os.path.exists(path):
            logger.warning(f"Feature importance file not found: {path}")
            return None
            
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading feature importance: {str(e)}")
        return None

# Page title
st.title("Global vs. Customer Model Comparison")
st.markdown("""
This page provides a detailed comparison between the global VAE model (trained on all legitimate transactions)
and the customer-specific VAE models (trained with transfer learning from the global model).
""")

# Load metrics
global_metrics = load_global_metrics()
customer_metrics = load_customer_metrics()
test_metrics = load_test_metrics()

# Check if data is available
if global_metrics is None or customer_metrics is None:
    st.error("Unable to load model metrics. Please run the training pipeline first.")
    st.stop()

# Sidebar for customer selection
st.sidebar.header("Customer Selection")
if customer_metrics is not None and len(customer_metrics) > 0:
    # Sort customers by metrics for more meaningful selection
    sorted_customers = customer_metrics.sort_values("transaction_count", ascending=False)
    
    # Create selection options: top N by transaction count and top N by fraud count
    top_by_txn = sorted_customers.head(10)["customer_id"].tolist()
    top_by_fraud = sorted_customers.sort_values("fraud_count", ascending=False).head(10)["customer_id"].tolist()
    
    # Combine and deduplicate
    featured_customers = list(set(top_by_txn + top_by_fraud))
    
    # Add option for statistics across all customers
    customer_options = ["All Customers"] + [f"Customer {cid}" for cid in featured_customers]
    selected_option = st.sidebar.selectbox("Select customer for comparison", customer_options)
    
    if selected_option == "All Customers":
        selected_customer = None
    else:
        selected_customer = int(selected_option.split(" ")[1])
else:
    st.sidebar.warning("No customer models available.")
    selected_customer = None

# Display metrics overview
st.header("Performance Metrics Overview")

# Metrics comparison
global_perf = global_metrics.get("performance", {})
test_global_perf = test_metrics.get("global_model", {}) if test_metrics else {}

metrics_to_display = [
    ("Recall", "recall"), 
    ("Specificity", "specificity"),
    ("Precision", "precision"),
    ("F1 Score", "f1_score"),
    ("ROC AUC", "roc_auc"),
    ("Optimal Threshold", "threshold")
]

# Create metrics table
metrics_data = []

# Global model metrics (train)
global_row = {"Model": "Global (Train)"}
for name, key in metrics_to_display:
    global_row[name] = f"{global_perf.get(key, 0):.4f}"
metrics_data.append(global_row)

# Global model metrics (test)
if test_global_perf:
    global_test_row = {"Model": "Global (Test)"}
    for name, key in metrics_to_display:
        global_test_row[name] = f"{test_global_perf.get(key, 0):.4f}"
    metrics_data.append(global_test_row)

# Individual customer metrics or average
if selected_customer is not None:
    # Single customer
    customer_row = {"Model": f"Customer {selected_customer}"}
    customer_data = customer_metrics[customer_metrics["customer_id"] == selected_customer]
    
    if len(customer_data) > 0:
        for name, key in metrics_to_display:
            customer_row[name] = f"{customer_data.iloc[0].get(key, 0):.4f}"
        metrics_data.append(customer_row)
    else:
        st.warning(f"No data found for Customer {selected_customer}")
else:
    # Average across all customers
    avg_row = {"Model": "Customer Average"}
    for name, key in metrics_to_display:
        if key in customer_metrics.columns:
            avg_row[name] = f"{customer_metrics[key].mean():.4f}"
        else:
            avg_row[name] = "N/A"
    metrics_data.append(avg_row)

# Create DataFrame for display
metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True)

# Performance curves
st.header("Performance Curves")

# Function to plot ROC and PR curves
def plot_curves(selected_customer=None):
    # Load curve data
    global_roc = load_curves("roc", "global")
    global_pr = load_curves("pr", "global")
    
    # Load test curves if available
    test_global_roc = None
    test_global_pr = None
    test_dir = Path("artifacts/evaluation/test/curves")
    if test_dir.exists():
        if (test_dir / "global_roc.csv").exists():
            test_global_roc = pd.read_csv(test_dir / "global_roc.csv")
        if (test_dir / "global_pr.csv").exists():
            test_global_pr = pd.read_csv(test_dir / "global_pr.csv")
    
    # Load customer curves if selected
    customer_roc = None
    customer_pr = None
    if selected_customer is not None:
        customer_roc = load_curves("roc", "customer", selected_customer)
        customer_pr = load_curves("pr", "customer", selected_customer)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot ROC curves
    if global_roc is not None:
        ax1.plot(global_roc["fpr"], global_roc["tpr"], label="Global (Train)", color="blue")
    if test_global_roc is not None:
        ax1.plot(test_global_roc["fpr"], test_global_roc["tpr"], label="Global (Test)", color="blue", linestyle="--")
    if customer_roc is not None:
        ax1.plot(customer_roc["fpr"], customer_roc["tpr"], label=f"Customer {selected_customer}", color="orange")
    
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot PR curves
    if global_pr is not None:
        ax2.plot(global_pr["recall"], global_pr["precision"], label="Global (Train)", color="blue")
    if test_global_pr is not None:
        ax2.plot(test_global_pr["recall"], test_global_pr["precision"], label="Global (Test)", color="blue", linestyle="--")
    if customer_pr is not None:
        ax2.plot(customer_pr["recall"], customer_pr["precision"], label=f"Customer {selected_customer}", color="orange")
    
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Display performance curves
st.pyplot(plot_curves(selected_customer))

# Display customer metrics distributions (if showing all customers)
if selected_customer is None and customer_metrics is not None and len(customer_metrics) > 1:
    st.header("Customer Models Performance Distribution")
    
    # Select metrics to visualize
    dist_metrics = ["recall", "specificity", "roc_auc", "precision"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(dist_metrics):
        if metric in customer_metrics.columns:
            # Create histogram with KDE
            sns.histplot(customer_metrics[metric], ax=axes[i], kde=True)
            
            # Add global model value as vertical line
            if metric in global_perf:
                global_value = global_perf[metric]
                axes[i].axvline(
                    x=global_value, 
                    color='red', 
                    linestyle='--', 
                    label=f'Global Model: {global_value:.3f}'
                )
                axes[i].legend()
            
            axes[i].set_title(f"Distribution of {metric.capitalize()}")
            axes[i].set_xlabel(metric.capitalize())
            axes[i].set_ylabel("Count")
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Customer metrics table (paginated)
    st.subheader("All Customer Models Metrics")
    page_size = 10
    total_pages = (len(customer_metrics) + page_size - 1) // page_size
    
    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    else:
        page = 1
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(customer_metrics))
    
    display_cols = ["customer_id", "transaction_count", "fraud_count", "recall", 
                    "specificity", "precision", "roc_auc"]
    
    st.dataframe(
        customer_metrics.iloc[start_idx:end_idx][display_cols].sort_values("customer_id"),
        use_container_width=True
    )

# Display individual customer details
elif selected_customer is not None:
    # Get customer data
    customer_data = customer_metrics[customer_metrics["customer_id"] == selected_customer]
    
    if len(customer_data) > 0:
        st.header(f"Customer {selected_customer} Details")
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transaction Count", f"{int(customer_data['transaction_count'].iloc[0]):,}")
        with col2:
            st.metric("Fraud Count", f"{int(customer_data['fraud_count'].iloc[0]):,}")
        with col3:
            fraud_rate = customer_data['fraud_count'].iloc[0] / customer_data['transaction_count'].iloc[0] * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        # Metrics comparison with global model
        st.subheader("Performance Comparison with Global Model")
        
        # Create comparison dataframe
        compare_metrics = ["recall", "specificity", "precision", "roc_auc"]
        comparison_data = []
        
        for metric in compare_metrics:
            if metric in global_perf and metric in customer_data.columns:
                global_val = global_perf[metric]
                customer_val = customer_data[metric].iloc[0]
                diff = customer_val - global_val
                diff_pct = (diff / global_val) * 100 if global_val != 0 else float('inf')
                
                comparison_data.append({
                    "Metric": metric.capitalize(),
                    "Global Model": f"{global_val:.4f}",
                    "Customer Model": f"{customer_val:.4f}",
                    "Difference": f"{diff:.4f}",
                    "Difference %": f"{diff_pct:+.2f}%"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Feature importance comparison (if available)
        global_importance = load_feature_importance("global")
        customer_importance = load_feature_importance("customer", selected_customer)
        
        if global_importance is not None and customer_importance is not None:
            st.subheader("Feature Importance Comparison")
            
            # Merge importance dataframes
            global_importance = global_importance.rename(
                columns={"importance": "global_importance"}
            ).set_index("feature")
            
            customer_importance = customer_importance.rename(
                columns={"importance": "customer_importance"}
            ).set_index("feature")
            
            # Join on feature
            combined = global_importance.join(
                customer_importance, how="outer"
            ).reset_index().fillna(0)
            
            # Sort by maximum importance
            combined["max_importance"] = combined[["global_importance", "customer_importance"]].max(axis=1)
            combined = combined.sort_values("max_importance", ascending=False).head(15)
            
            # Plot side by side
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(combined))
            width = 0.35
            
            ax.barh(x - width/2, combined["global_importance"], width, label="Global Model", color="blue", alpha=0.7)
            ax.barh(x + width/2, combined["customer_importance"], width, label="Customer Model", color="orange", alpha=0.7)
            
            ax.set_yticks(x)
            ax.set_yticklabels(combined["feature"])
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"Top Features: Global vs Customer {selected_customer}")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="x")
            
            plt.tight_layout()
            st.pyplot(fig)

# Display Test vs. Train Performance (if test metrics available)
if test_metrics is not None:
    st.header("Test vs. Train Performance")
    
    st.markdown("""
    This section compares model performance on training data versus test data.
    The test dataset represents unseen transactions that weren't used during model training.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Global Model")
        
        # Compile performance metrics
        test_vs_train = []
        for name, key in metrics_to_display:
            if key in global_perf and key in test_global_perf:
                train_val = global_perf[key]
                test_val = test_global_perf[key]
                diff = test_val - train_val
                diff_pct = (diff / train_val) * 100 if train_val != 0 else 0
                
                test_vs_train.append({
                    "Metric": name,
                    "Train": f"{train_val:.4f}",
                    "Test": f"{test_val:.4f}",
                    "Difference": f"{diff:+.4f}",
                    "Difference %": f"{diff_pct:+.2f}%"
                })
        
        test_train_df = pd.DataFrame(test_vs_train)
        st.dataframe(test_train_df, use_container_width=True)
    
    with col2:
        # Get generalization ratio
        st.subheader("Generalization Performance")
        
        if "roc_auc" in global_perf and "roc_auc" in test_global_perf:
            gen_ratio = test_global_perf["roc_auc"] / global_perf["roc_auc"]
            
            # Create gauge chart
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh([0], [gen_ratio], height=0.5, color="green" if gen_ratio >= 0.9 else "orange" if gen_ratio >= 0.7 else "red")
            ax.barh([0], [1.0], height=0.5, color="gray", alpha=0.3)
            ax.set_xlim(0, 1.1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.axvline(x=1.0, color="blue", linestyle="--", alpha=0.7)
            ax.set_title(f"Generalization Ratio: {gen_ratio:.2f}")
            
            # Add labels
            ax.text(0.1, 0, "Poor", verticalalignment="center")
            ax.text(0.6, 0, "Good", verticalalignment="center")
            ax.text(1.0, 0, "Excellent", verticalalignment="center", horizontalalignment="center")
            
            st.pyplot(fig)
            
            st.markdown(f"""
            **Generalization Ratio:** {gen_ratio:.2f}
            
            This ratio represents how well the model performs on unseen data compared to training data:
            - **< 0.7**: Potential overfitting
            - **0.7-0.9**: Good generalization
            - **> 0.9**: Excellent generalization
            - **> 1.0**: Model performs better on test data than training data
            """)

# Footer with additional information
st.divider()
st.markdown("""
    ### Why Customer-Specific Models?
    
    The global VAE model learns general fraud patterns across all customers, while customer-specific models 
    adapt to individual transaction behaviors. This dual-model approach addresses the challenge of balancing 
    high recall (catching all fraud) with reasonable specificity (minimizing false positives).
    
    Customer-specific models are particularly valuable because:
    1. Geographic patterns vary widely between customers but are consistent for individuals
    2. Transaction timing and amount patterns are often customer-specific
    3. Device and email usage patterns differ between customers
    
    The transfer learning approach ensures that customer models benefit from the global model's knowledge
    while still adapting to individual behavior patterns.
""")