"""
Comparison page: side-by-side metrics & curves for global vs. customers.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Global vs Customer Models")

# Load
gm = pd.read_json("metrics/global_metrics.json", typ="series")
cm = pd.read_csv("metrics/customers/customer_metrics.csv")

st.subheader("Key Metrics")
st.markdown(f"**Global** &mdash; Recall: {gm['recall']:.3f}, Spec: {gm['specificity']:.3f}, AUC: {gm['roc_auc']:.3f}")
st.markdown(f"**Customer (mean)** &mdash; Recall: {cm['recall'].mean():.3f}, Spec: {cm['specificity'].mean():.3f}, AUC: {cm['roc_auc'].mean():.3f}")

# Curves (sample one customer)
roc_g = pd.read_csv("metrics/global_roc.csv")
pr_g  = pd.read_csv("metrics/global_pr.csv")
cust_id = cm["card1"].iloc[0]
roc_c = pd.read_csv(f"metrics/customers/{cust_id}_roc.csv")
pr_c  = pd.read_csv(f"metrics/customers/{cust_id}_pr.csv")

fig, axes = plt.subplots(2,2, figsize=(12,10))
axes[0,0].plot(roc_g["fpr"],roc_g["tpr"],label="Global"); axes[0,0].set_title("Global ROC")
axes[0,1].plot(pr_g["recall"],pr_g["precision"],label="Global"); axes[0,1].set_title("Global PR")
axes[1,0].plot(roc_c["fpr"],roc_c["tpr"],label="Customer"); axes[1,0].set_title(f"Cust {cust_id} ROC")
axes[1,1].plot(pr_c["recall"],pr_c["precision"],label="Customer"); axes[1,1].set_title(f"Cust {cust_id} PR")
for ax in axes.flatten():
    ax.legend(); ax.grid(True)

st.pyplot(fig)