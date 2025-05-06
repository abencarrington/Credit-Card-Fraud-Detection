"""
Enhanced main Streamlit app for the IEEE-CIS fraud detection dashboard.

Features:
- Improved UI layout and styling
- Data loading state management
- Model performance overview
- Interactive navigation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.loader import load_data
from src.data.processor import preprocess_data, extract_time_features
from src.features.global_features import add_global_features
from src.features.customer_features import add_customer_features
from src.bootstrap import set_project_root

# Set project root to resolve paths correctly
set_project_root()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="VAE Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🛡️ Dual-Model VAE Fraud Detection")
st.markdown("""
This dashboard presents a sophisticated credit card fraud detection system that leverages Variational Autoencoders (VAEs) 
with both global and customer-specific models. The system is trained on the IEEE-CIS Fraud Detection dataset.
""")

# Function to load metrics
@st.cache_data
def load_metrics(metrics_path="artifacts/metrics/global_metrics.json"):
    """Load model metrics from JSON file."""
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        return None

# Function to load model performance data
@st.cache_data
def load_performance_data():
    """Load and prepare model performance data for visualization."""
    global_metrics = load_metrics()
    if not global_metrics:
        return None
    
    # Load customer metrics if available
    customer_metrics_path = "artifacts/metrics/customer_metrics_summary.csv"
    try:
        customer_df = pd.read_csv(customer_metrics_path)
        has_customer_metrics = True
    except Exception:
        has_customer_metrics = False
    
    return {
        "global": global_metrics,
        "customers": customer_df if has_customer_metrics else None,
        "has_customer_metrics": has_customer_metrics
    }

# Sidebar for navigation and data loading
with st.sidebar:
    st.header("Navigation")
    st.markdown("Use the pages menu at left to navigate between dashboard sections.")
    
    st.divider()
    
    st.header("Data & Models")
    data_dir = st.text_input("Data folder", "data/raw")
    
    data_loading_container = st.container()
    
    # Data loading button
    if st.button("Load & Preview Data"):
        with st.spinner("Loading data..."):
            with data_loading_container:
                try:
                    # Try to load data
                    df = load_data(data_dir, train=True)
                    
                    # Show loading success message
                    st.success(f"✅ Loaded {len(df):,} transactions!")
                    
                    # Display metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Transactions", f"{len(df):,}")
                    with col2:
                        fraud_count = df['isFraud'].sum() if 'isFraud' in df.columns else 0
                        fraud_percent = fraud_count / len(df) * 100 if len(df) > 0 else 0
                        st.metric("Fraud Transactions", f"{fraud_count:,} ({fraud_percent:.2f}%)")
                    
                    # Show sample of data
                    st.write("### Sample transactions")
                    st.dataframe(df.sample(5), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.info("Please make sure the IEEE-CIS dataset files are in the specified directory.")

# Load performance metrics for main page
performance_data = load_performance_data()

# Main page content - Model Overview
if performance_data:
    # Display summary metrics
    st.header("Model Performance Overview")
    
    metrics = performance_data["global"]["performance"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fraud Recall", f"{metrics['recall']:.3f}")
    with col2:
        st.metric("Specificity", f"{metrics['specificity']:.3f}")
    with col3:
        st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    with col4:
        st.metric("F-Beta (β=30)", f"{metrics.get('f_beta', metrics.get('fbeta_score', 0)):.3f}")
    
    # Display visualization
    st.subheader("Global Model Performance")
    
    try:
        # Try to load visualization from assets
        st.image("streamlit/assets/global_roc_pr.png", 
                caption="Global Model ROC and Precision-Recall Curves",
                use_column_width=True)
    except Exception:
        # If image not found, display a message
        st.info("ROC and PR curve visualization not found. Run the full pipeline first to generate visualizations.")
        
    # Customer models section
    st.subheader("Customer-Specific Models")
    
    if performance_data["has_customer_metrics"]:
        try:
            # Try to load visualization from assets
            st.image("streamlit/assets/customer_performance.png", 
                    caption="Customer Models Performance Distribution",
                    use_column_width=True)
        except Exception:
            # If image not found, display metrics in a table
            customer_df = performance_data["customers"]
            st.write("#### Customer Models Summary Statistics")
            st.dataframe(customer_df.describe().round(3).T, use_container_width=True)
    else:
        st.info("Customer model metrics not available. Run the full pipeline to train customer models.")
else:
    # If metrics not found, show instructions
    st.info("""
    ### Getting Started
    
    1. Use the sidebar to load and preview data
    2. Navigate to different pages using the menu on the left
    3. Run the full pipeline with `doit` to generate model metrics and visualizations
    
    ```bash
    # Run the full pipeline
    python -m doit
    ```
    """)

# Footer with additional information
st.divider()
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em;">
    IEEE-CIS Fraud Detection with Variational Autoencoders | Data last processed: April 2025
</div>
""", unsafe_allow_html=True)