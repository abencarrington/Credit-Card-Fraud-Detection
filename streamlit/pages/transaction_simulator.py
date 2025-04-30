"""
Transaction Simulator: allows setting key fields, then computes
anomaly score via the trained global VAE’s reconstruction error.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from src.training.train_global import train_global
from src.data.processor   import preprocess_data
from src.features.global_features import add_global_features

st.title("Transaction Simulator")

@st.cache(allow_output_mutation=True)
def load_model():
    """
    Train (or load) the global VAE once and return: encoder, decoder, threshold.
    """
    enc, dec, metrics = train_global(
        data_dir="./data/raw",
        out_dir="./metrics",
        epochs=30  # shorter for interactivity
    )
    return enc, dec, metrics["threshold"]

enc, dec, base_thr = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.form("simulator"):
    st.write("**Set transaction features**")
    amt    = st.number_input("Amount", min_value=0.0, value=100.0)
    hour   = st.slider("Hour of day", 0,23,12)
    day    = st.slider("Day of month",1,31,15)
    dow    = st.selectbox("Day of week", list(range(7)), index=3)
    card1  = st.text_input("Card1 ID", "1000")
    device = st.selectbox("DeviceType", ["desktop","mobile","unknown"])
    p_email = st.selectbox("P_emaildomain", ["gmail.com","yahoo.com","unknown"])
    submitted = st.form_submit_button("Score Transaction")

if submitted:
    # Build a single-row DataFrame
    df_sim = pd.DataFrame([{
        "TransactionDT": 0,  # dummy, will be overwritten
        "TransactionAmt": amt,
        "card1": int(card1),
        "DeviceType": device,
        "P_emaildomain": p_email
    }])
    # apply features + preprocess
    df_sim = add_global_features(df_sim)
    df_sim = preprocess_data(df_sim)

    # to tensor
    X = torch.tensor(df_sim.to_numpy(dtype=np.float32), device=device)
    # forward pass
    mu, lv = enc(X)
    s  = torch.exp(0.5*lv)+1e-7
    z  = dist.Normal(mu,s).sample()
    mu2, lv2 = dec(z)
    s2 = torch.exp(0.5*lv2)+1e-7
    err = -dist.Normal(mu2,s2).log_prob(X).sum(-1).item()

    # anomaly score
    score = (err - base_thr) / (err + base_thr)
    st.metric("Reconstruction error", f"{err:.2f}")
    st.metric("Anomaly score", f"{score:.3f}")

    if err >= base_thr:
        st.error("⚠️ Flagged as Fraud")
    else:
        st.success("✅ Likely Legitimate")