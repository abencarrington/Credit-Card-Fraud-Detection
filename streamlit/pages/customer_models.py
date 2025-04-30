"""
Customer models page: train & evaluate VAEs per customer,
show distribution of their performance.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.training.train_customer import train_all_customers

st.title("Customer-Specific VAE Models")
if st.button("Train & Evaluate All Customer VAEs"):
    rows = train_all_customers()
    df = pd.DataFrame(rows)

    st.subheader("Performance summary")
    st.dataframe(df.describe().T)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    df["recall"].hist(ax=ax1, bins=20)
    ax1.set(title="Recall distribution", xlabel="Recall")
    df["specificity"].hist(ax=ax2, bins=20)
    ax2.set(title="Specificity distribution", xlabel="Specificity")
    st.pyplot(fig)