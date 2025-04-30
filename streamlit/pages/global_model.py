"""
Global model page: train/evaluate the global VAE on all data,
display metrics, curves, and SHAP summary.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.training.train_global import train_global

st.title("Global VAE Model")
if st.button("Train & Evaluate Global VAE"):
    enc, dec, metrics = train_global()
    st.subheader("Metrics")
    st.json(metrics)

    roc_df = pd.read_csv("metrics/global_roc.csv")
    pr_df  = pd.read_csv("metrics/global_pr.csv")

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    ax1.plot(pr_df["recall"], pr_df["precision"], label="PR")
    ax1.set( title="Precision-Recall", xlabel="Recall", ylabel="Precision" )
    ax2.plot(roc_df["fpr"], roc_df["tpr"], label="ROC")
    ax2.set( title="ROC Curve", xlabel="FPR", ylabel="TPR" )
    st.pyplot(fig)

    st.image("metrics/global_shap.png", caption="Global Model SHAP Summary")