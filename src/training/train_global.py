"""
Train and evaluate the global VAE on ALL legitimate transactions.
Saves metrics, PR/ROC curves, and a SHAP summary plot.
"""

import os
import json
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import shap
import torch
import pyro
import pyro.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ReduceLROnPlateau
from sklearn.metrics import (
    recall_score, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)

from src.data.loader    import load_data
from src.data.processor import preprocess_data
from src.features.global_features import add_global_features
from src.models.vae     import Encoder, Decoder
from src.evaluation.metrics import plot_pr_roc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_global(
    data_dir: str = "./data/raw",
    out_dir:  str = "./metrics",
    z_dim:    int = 2,
    h1:       int = 128,
    h2:       int = 64,
    lr:    float=1e-4,
    batch_size:int=128,
    epochs:    int=50
):
    """
    Load, preprocess, train VAE, evaluate, and save metrics/plots.

    Returns:
        encoder, decoder, metrics dict
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=== GLOBAL TRAINING START ===")

    # Load & prep
    df = load_data(data_dir, train=True)
    df = add_global_features(df)
    df = preprocess_data(df)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Keep only legit for training
    X_train = X_train[y==0].to_numpy(dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    enc = Encoder(X_train.shape[1], h1, h2, z_dim).to(device)
    dec = Decoder(z_dim, h2, h1, X_train.shape[1]).to(device)

    # SVI setup
    def model(x):
        pyro.module("decoder", dec)
        with pyro.plate("data", x.shape[0]):
            z = pyro.sample("z",
                dist.Normal(0.,1.)
                    .expand([x.shape[0],z_dim])
                    .to_event(1)
            )
            mu, lv = dec(z)
            s = torch.exp(0.5*lv)+1e-7
            pyro.sample("obs", dist.Normal(mu,s).to_event(1), obs=x)

    def guide(x):
        pyro.module("encoder", enc)
        with pyro.plate("data", x.shape[0]):
            mu, lv = enc(x)
            s = torch.exp(0.5*lv)+1e-7
            pyro.sample("z", dist.Normal(mu,s).to_event(1))

    svi = SVI(
        model, guide,
        ReduceLROnPlateau({
            "optimizer":torch.optim.Adam,
            "optim_args":{"lr":lr},
            "factor":0.5,"patience":5,
            "threshold":1e-4,"threshold_mode":"rel"
        }),
        loss=Trace_ELBO()
    )

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).to(device)),
        batch_size=batch_size,
        shuffle=True
    )

    # Train loop
    logger.info("Training global VAE…")
    for epoch in tqdm(range(epochs), desc="Epoch"):
        total_loss = 0.0
        for batch, in train_loader:
            total_loss += svi.step(batch)
        avg = total_loss/len(X_train)
        logger.debug(f"Epoch {epoch} → loss {avg:.4f}")

    # Evaluate: reconstruction errors
    X_test_np = X_test.to_numpy(dtype=np.float32)
    X_test_t  = torch.from_numpy(X_test_np).to(device)
    recon = []
    with torch.no_grad():
        for i in range(0, len(X_test_np), batch_size):
            xb = X_test_t[i:i+batch_size]
            mu, lv = enc(xb)
            s  = torch.exp(0.5*lv)+1e-7
            z  = dist.Normal(mu,s).sample()
            mu2, lv2 = dec(z)
            s2 = torch.exp(0.5*lv2)+1e-7
            err = -dist.Normal(mu2,s2).log_prob(xb)
            recon.extend(err.sum(-1).cpu().numpy())

    # Plot & threshold
    y_true = y_test.values
    thr = plot_pr_roc(y_true, np.array(recon), beta=30)

    # Compute final metrics
    y_pred = (np.array(recon) >= thr).astype(int)
    rec = recall_score(y_true,y_pred,pos_label=1)
    tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
    spec = tn/(tn+fp)
    fpr,tpr,_ = roc_curve(y_true, np.array(recon))
    roc_auc = auc(fpr,tpr)

    metrics = {
        "recall": rec,
        "specificity": spec,
        "roc_auc":    roc_auc,
        "threshold":  thr
    }

    # Save metrics & curves
    with open(os.path.join(out_dir,"global_metrics.json"), "w") as f:
        json.dump(metrics, f)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(
        os.path.join(out_dir,"global_roc.csv"), index=False
    )
    pr, re, _ = precision_recall_curve(y_true,np.array(recon))
    pd.DataFrame({"precision":pr,"recall":re}).to_csv(
        os.path.join(out_dir,"global_pr.csv"), index=False
    )

    # SHAP
    logger.info("Computing SHAP summary…")
    bg = torch.tensor(X_train[np.random.choice(len(X_train),200,replace=False)]).to(device)
    explainer = shap.DeepExplainer(lambda x: dec(enc(x)[0])[0], bg)
    sv = explainer.shap_values(torch.tensor(X_test_np[:200]).to(device))
    shap.summary_plot(sv, X_test_np[:200], show=False)
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(out_dir,"global_shap.png"), bbox_inches="tight")

    logger.info("=== GLOBAL TRAINING COMPLETE ===")
    return enc, dec, metrics