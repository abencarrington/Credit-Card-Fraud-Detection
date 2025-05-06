"""
Transaction Simulator: Interactive dashboard for testing the VAE fraud detection system.

Features:
1. Dynamic form based on feature distributions
2. Visual anomaly score explanation
3. Comparative performance: global vs customer-specific models
4. Feature importance visualization
5. Transaction history tracking
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import shap

from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

from src.models.vae import GlobalVAE, CustomerVAE
from src.data.processor import preprocess_data
from src.features.global_features import add_global_features
from src.features.customer_features import add_customer_features

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Transaction Simulator", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache loaders for efficiency
@st.cache_resource
def load_global_model(model_path: str = "models/global_vae.pt") -> GlobalVAE:
    """Load the global VAE model."""
    try:
        # Get input dimension from saved metrics
        with open("artifacts/global_metrics.json", "r") as f:
            metrics = json.load(f)
            input_dim = metrics.get("model_params", {}).get("input_dim", 100)
        
        # Load model
        model = GlobalVAE.load(model_path, input_dim=input_dim)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading global model: {str(e)}")
        # Create dummy model for demo if needed
        return GlobalVAE(input_dim=100, hidden1=64, hidden2=32, zdim=2)


@st.cache_resource
def load_customer_model(customer_id: int, model_dir: str = "models/customers") -> Optional[CustomerVAE]:
    """Load a customer-specific VAE model."""
    try:
        model_path = os.path.join(model_dir, f"customer_{customer_id}.pt")
        if os.path.exists(model_path):
            # Get input dimension from saved metrics
            with open("artifacts/global_metrics.json", "r") as f:
                metrics = json.load(f)
                input_dim = metrics.get("model_params", {}).get("input_dim", 100)
            
            # Load model
            model = CustomerVAE.load(model_path, input_dim=input_dim)
            model.eval()
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading customer model: {str(e)}")
        return None


@st.cache_data
def load_feature_distributions(data_path: str = "data/processed/feature_distributions.json") -> Dict:
    """Load feature distributions for realistic form values."""
    try:
        with open(data_path, "r") as f:
            return json.load(f)
    except Exception as e:
        # Return default distributions if file not found
        logger.warning(f"Feature distributions file not found: {str(e)}")
        return {
            "TransactionAmt": {"min": 1.0, "max": 10000.0, "median": 100.0},
            "hour": {"min": 0, "max": 23, "median": 12},
            "day": {"min": 1, "max": 31, "median": 15},
            "dow": {"values": list(range(7)), "frequencies": [1/7] * 7},
            "card1": {"values": list(range(1000, 10000, 1000)), "frequencies": [1/9] * 9},
            "DeviceType": {"values": ["desktop", "mobile", "tablet"], "frequencies": [0.6, 0.3, 0.1]},
            "P_emaildomain": {
                "values": ["gmail.com", "yahoo.com", "hotmail.com", "aol.com", "unknown"],
                "frequencies": [0.4, 0.3, 0.15, 0.1, 0.05]
            }
        }


@st.cache_data
def load_threshold(metrics_path: str = "artifacts/global_metrics.json") -> float:
    """Load the optimal threshold for anomaly detection."""
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            return metrics.get("performance", {}).get("threshold", 50.0)
    except Exception as e:
        logger.warning(f"Error loading threshold: {str(e)}")
        return 50.0


# Initialize session state for transaction history
if "transaction_history" not in st.session_state:
    st.session_state.transaction_history = []

# Main layout
st.title("🔍 Transaction Simulator")
st.markdown("""
This simulator allows you to create custom transaction scenarios and test the fraud detection system in real-time.
The system uses a Variational Autoencoder (VAE) trained exclusively on legitimate transactions to detect anomalies.
""")

# Load resources
distributions = load_feature_distributions()
global_threshold = load_threshold()
global_model = load_global_model()

# Sidebar for feature selection
st.sidebar.header("Transaction Features")

with st.sidebar.form("transaction_form"):
    st.subheader("Set Transaction Parameters")
    
    # Transaction amount with distribution-aware defaults
    amt_dist = distributions.get("TransactionAmt", {"min": 1.0, "max": 10000.0, "median": 100.0})
    amt = st.number_input(
        "Transaction Amount ($)",
        min_value=float(amt_dist.get("min", 1.0)),
        max_value=float(amt_dist.get("max", 10000.0)),
        value=float(amt_dist.get("median", 100.0)),
        step=10.0
    )
    
    # Time-based features
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider(
            "Hour of Day", 
            min_value=0, 
            max_value=23, 
            value=distributions.get("hour", {}).get("median", 12)
        )
    with col2:
        day = st.slider(
            "Day of Month",
            min_value=1,
            max_value=31,
            value=distributions.get("day", {}).get("median", 15)
        )
    
    dow_options = distributions.get("dow", {}).get("values", list(range(7)))
    dow_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_map = {i: label for i, label in enumerate(dow_labels)}
    dow = st.selectbox(
        "Day of Week",
        options=dow_options,
        format_func=lambda x: dow_map.get(x, str(x)),
        index=3  # Default to Thursday
    )
    
    # Card and device features
    col1, col2 = st.columns(2)
    with col1:
        card_options = distributions.get("card1", {}).get("values", list(range(1000, 10000, 1000)))
        card1 = st.selectbox(
            "Card ID",
            options=card_options,
            index=0
        )
    with col2:
        device_options = distributions.get("DeviceType", {}).get("values", ["desktop", "mobile", "tablet"])
        device = st.selectbox(
            "Device Type",
            options=device_options,
            index=0
        )
    
    # Email domain
    email_options = distributions.get("P_emaildomain", {}).get(
        "values", ["gmail.com", "yahoo.com", "hotmail.com", "aol.com", "unknown"]
    )
    p_email = st.selectbox(
        "Email Domain",
        options=email_options,
        index=0
    )
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        use_customer_model = st.checkbox(
            "Use Customer-Specific Model",
            value=True,
            help="If checked, will use both global and customer-specific models for comparison."
        )
        
        visualize_shap = st.checkbox(
            "Visualize Feature Importance",
            value=True,
            help="If checked, will show SHAP values to explain the model's decision."
        )
    
    # Submit button
    submitted = st.form_submit_button("Score Transaction")

# Main content area
if submitted:
    # Start analysis process
    with st.spinner("Analyzing transaction..."):
        # Build a single-row DataFrame
        transaction_time = datetime.now()
        df_sim = pd.DataFrame([{
            "TransactionDT": int(transaction_time.timestamp()),
            "TransactionAmt": amt,
            "card1": int(card1),
            "DeviceType": device,
            "P_emaildomain": p_email,
            "hour": hour,
            "day": day,
            "dow": dow
        }])
        
        # Apply features + preprocess
        df_sim = add_global_features(df_sim)
        if use_customer_model:
            df_sim = add_customer_features(df_sim)
        df_sim = preprocess_data(df_sim)
        
        # Convert to tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(df_sim.values.astype(np.float32), device=device)
        
        # Get global model prediction
        global_model.eval()
        with torch.no_grad():
            global_error = global_model.compute_reconstruction_error(X).cpu().item()
            
            # Generate decoded output for visualization
            global_recon = global_model.reconstruct(X).cpu().numpy()
        
        # Calculate anomaly score (normalize between 0-1)
        global_score = min(max((global_error - global_threshold) / (global_error + global_threshold), 0.0), 1.0)
        
        # Get customer model prediction if available and requested
        customer_error = None
        customer_score = None
        customer_recon = None
        customer_model = None
        
        if use_customer_model:
            customer_model = load_customer_model(int(card1))
            if customer_model is not None:
                customer_model.eval()
                with torch.no_grad():
                    customer_error = customer_model.compute_reconstruction_error(X).cpu().item()
                    customer_recon = customer_model.reconstruct(X).cpu().numpy()
                
                # Get customer threshold from metrics
                customer_threshold = global_threshold  # Default fallback
                try:
                    # Try to load customer-specific threshold
                    customer_metrics_path = f"artifacts/customers/customer_{int(card1)}_metrics.json"
                    if os.path.exists(customer_metrics_path):
                        with open(customer_metrics_path, "r") as f:
                            customer_metrics = json.load(f)
                            customer_threshold = customer_metrics.get(
                                "performance", {}).get("threshold", global_threshold)
                except Exception as e:
                    logger.warning(f"Error loading customer threshold: {str(e)}")
                
                # Calculate customer anomaly score
                customer_score = min(max((customer_error - customer_threshold) / 
                                        (customer_error + customer_threshold), 0.0), 1.0)
        
        # Store in transaction history
        transaction_record = {
            "timestamp": transaction_time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amt,
            "card_id": int(card1),
            "device": device,
            "email": p_email,
            "global_score": global_score,
            "global_error": global_error,
            "is_fraud_global": global_error >= global_threshold,
            "customer_score": customer_score,
            "customer_error": customer_error,
            "is_fraud_customer": (customer_error >= global_threshold 
                                if customer_error is not None else None)
        }
        st.session_state.transaction_history.append(transaction_record)
    
    # Display results
    st.markdown("## Transaction Analysis Results")
    
    # Split into columns for global and customer models
    if use_customer_model and customer_model is not None:
        col1, col2 = st.columns(2)
    else:
        col1, col2 = st.columns([2, 1])
    
    # Global model results
    with col1:
        st.markdown("### Global Model Assessment")
        
        # Display anomaly score with gauge visualization
        st.markdown(f"**Reconstruction Error:** {global_error:.2f}")
        
        # Create colored progress bar for anomaly score
        score_color = "green"
        if global_score > 0.7:
            score_color = "red"
        elif global_score > 0.4:
            score_color = "orange"
        
        st.markdown(f"**Anomaly Score:** {global_score:.3f}")
        st.progress(float(global_score), text=None)
        
        # Final classification
        if global_error >= global_threshold:
            st.error("⚠️ **FLAGGED AS POTENTIALLY FRAUDULENT**")
        else:
            st.success("✅ **LIKELY LEGITIMATE TRANSACTION**")
        
        # SHAP visualization if requested
        if visualize_shap:
            st.markdown("#### Feature Importance")
            try:
                # Use a single example for SHAP
                background = torch.zeros((1, X.shape[1]), device=device)
                explainer = shap.DeepExplainer(
                    lambda x: global_model.reconstruct(x), 
                    background
                )
                shap_values = explainer.shap_values(X)
                
                # Create force plot
                fig, ax = plt.subplots(figsize=(10, 3))
                shap.force_plot(
                    base_value=explainer.expected_value[0],
                    shap_values=shap_values[0][0],
                    features=df_sim.iloc[0],
                    feature_names=df_sim.columns,
                    matplotlib=True,
                    show=False,
                    axis_color='#333333',
                    text_rotation=45
                )
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Could not generate SHAP visualization: {str(e)}")
                
                # Show top features with highest difference between original and reconstructed
                feature_diff = np.abs(df_sim.values - global_recon)[0]
                feature_importance = pd.DataFrame({
                    'Feature': df_sim.columns,
                    'Importance': feature_diff
                }).sort_values('Importance', ascending=False)
                
                # Bar chart of top 10 features
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(
                    x='Importance', y='Feature', 
                    data=feature_importance.head(10),
                    ax=ax
                )
                ax.set_title('Top Features Contributing to Anomaly Score')
                st.pyplot(fig)
                plt.close()
    
    # Customer model results (if available)
    if use_customer_model and customer_model is not None and customer_error is not None:
        with col2:
            st.markdown("### Customer-Specific Model")
            
            # Display anomaly score
            st.markdown(f"**Reconstruction Error:** {customer_error:.2f}")
            
            # Colored progress bar
            c_score_color = "green"
            if customer_score > 0.7:
                c_score_color = "red"
            elif customer_score > 0.4:
                c_score_color = "orange"
            
            st.markdown(f"**Anomaly Score:** {customer_score:.3f}")
            st.progress(float(customer_score), text=None)
            
            # Final classification
            if customer_error >= global_threshold:
                st.error("⚠️ **FLAGGED AS POTENTIALLY FRAUDULENT**")
            else:
                st.success("✅ **LIKELY LEGITIMATE TRANSACTION**")
            
            # Model comparison
            st.markdown("#### Model Comparison")
            comparison_df = pd.DataFrame({
                'Model': ['Global', 'Customer'],
                'Anomaly Score': [global_score, customer_score],
                'Reconstruction Error': [global_error, customer_error]
            })
            st.bar_chart(comparison_df.set_index('Model'))
    
    # Additional details in an expander
    with st.expander("Transaction Details"):
        # Original vs reconstructed values
        if global_recon is not None:
            st.markdown("#### Original vs. Reconstructed Values")
            
            # Select top different features
            feature_diff = np.abs(df_sim.values - global_recon)[0]
            top_indices = np.argsort(-feature_diff)[:10]  # Top 10 most different
            
            # Create DataFrame for comparison
            compare_df = pd.DataFrame({
                'Feature': df_sim.columns[top_indices],
                'Original': df_sim.values[0, top_indices],
                'Reconstructed': global_recon[0, top_indices],
                'Difference': feature_diff[top_indices]
            })
            
            st.dataframe(compare_df.style.highlight_max(axis=0, subset=['Difference']))
    
    # Historical context
    st.markdown("## Transaction History")
    history_df = pd.DataFrame(st.session_state.transaction_history)
    if not history_df.empty:
        history_df = history_df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        # Add icons for fraud/legitimate
        def format_fraud_status(is_fraud):
            if is_fraud is None:
                return "N/A"
            return "⚠️ Fraud" if is_fraud else "✅ Legitimate"
        
        if 'is_fraud_global' in history_df.columns:
            history_df['Global Status'] = history_df['is_fraud_global'].apply(format_fraud_status)
        
        if 'is_fraud_customer' in history_df.columns:
            history_df['Customer Status'] = history_df['is_fraud_customer'].apply(format_fraud_status)
        
        # Select columns to display
        display_cols = ['timestamp', 'amount', 'card_id', 'Global Status']
        if 'Customer Status' in history_df.columns:
            display_cols.append('Customer Status')
        
        st.dataframe(
            history_df[display_cols].rename(columns={
                'timestamp': 'Time',
                'amount': 'Amount ($)',
                'card_id': 'Card ID'
            }),
            use_container_width=True
        )
        
        if len(history_df) > 1:
            st.markdown("### Transaction Trends")
            # Plot trend of anomaly scores over time
            fig, ax = plt.subplots(figsize=(10, 5))
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            plt.plot(
                history_df['timestamp'], 
                history_df['global_score'], 
                'b-o', 
                label='Global Model'
            )
            
            if 'customer_score' in history_df.columns:
                valid_mask = ~history_df['customer_score'].isna()
                if valid_mask.any():
                    plt.plot(
                        history_df.loc[valid_mask, 'timestamp'],
                        history_df.loc[valid_mask, 'customer_score'],
                        'r-o',
                        label='Customer Model'
                    )
            
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            plt.ylabel('Anomaly Score')
            plt.title('Anomaly Score Trend')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No transactions recorded yet.")

# Instructions at the bottom
with st.expander("How to Use This Simulator"):
    st.markdown("""
    ### Using the Transaction Simulator
    
    1. **Set Transaction Parameters**: Adjust the sliders and selectors in the sidebar to create different transaction scenarios.
    
    2. **Score Transaction**: Click the "Score Transaction" button to analyze the transaction using the VAE models.
    
    3. **Interpret Results**:
       - **Reconstruction Error**: The raw error between the original transaction and its reconstruction by the VAE.
       - **Anomaly Score**: Normalized score between 0-1, where higher values indicate more anomalous transactions.
       - **Feature Importance**: Shows which features contributed most to the anomaly detection.
    
    4. **Compare Models**: If available, you can compare the global model (trained on all data) with the customer-specific model.
    
    5. **Examine History**: View previous transactions and observe trends in the anomaly scores.
    """)

    st.markdown("""
    ### Tips for Testing Different Fraud Scenarios
    
    1. **Unusual Amount**: Try very large transaction amounts that deviate from normal patterns.
    
    2. **Time Anomalies**: Transactions at unusual hours (e.g., 3 AM) may trigger higher anomaly scores.
    
    3. **Changed Email/Device**: Using different email domains or devices than the customer typically uses.
    
    4. **Combination Factors**: Combine multiple unusual elements for more likely fraud detection.
    """)