"""
Main Streamlit app: sidebar for data loading & overview,
links to model pages in the sidebar.
"""

import logging
import streamlit as st
from src.data.loader      import load_data
from src.data.processor   import preprocess_data
from src.features.global_features  import add_global_features
from src.features.customer_features import add_customer_features

from src.bootstrap import set_project_root
set_project_root()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="VAE Fraud Dashboard", layout="wide")
st.title("Dual-Model VAE Fraud Detection")

st.sidebar.header("Data & Preview")
DATA_DIR = st.sidebar.text_input("Data folder", "data/raw")
if st.sidebar.button("Load & Preview"):
    df = load_data(DATA_DIR, train=True)
    df = add_global_features(df)
    df = add_customer_features(df)
    df = preprocess_data(df)
    st.write("### Sample of processed data")
    st.dataframe(df.sample(10))