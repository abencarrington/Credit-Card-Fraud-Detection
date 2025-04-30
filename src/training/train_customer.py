"""
Train a VAE per customer, save per-customer metrics and PR/ROC curves,
and aggregate them into customer_metrics.csv.
"""

import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
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

from src.data.loader     import load_data
from src.data.processor  import preprocess_data
from src.features.customer_features import add_customer_features
from src.models.vae      import Encoder, Decoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_all_customers(
    data_dir: str = "./data/raw",
    out_dir:  str = "./metrics/customers",
    z_dim:    int = 2,
    h1:       int = 64,
    h2:       int = 32,
    lr:    float=1e-4,
    epochs:   int=50,
    batch_size:int=64
):
    """
    For each card1 with ≥100 txns:
      - train a per-customer VAE
      - compute recall, specificity, ROC AUC
      - save that customer’s curves and metrics

    Returns:
        List of per-customer metrics dicts
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=== CUSTOMER TRAINING START ===")

    df = load_data(data_dir, train=True)
    df = add_customer_features(df)
    df = preprocess_data(df)

    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    card_ids = df["card1"].unique()

    for cust in tqdm(card_ids, desc="Customers"):
        sub = df[df["card1"] == cust]
        if len(sub) < 100:
            continue

        X = sub.drop("isFraud",axis=1).to_numpy(dtype=np.float32)
        y = sub["isFraud"].values

        enc = Encoder(X.shape[1],h1,h2,z_dim).to(device)
        dec = Decoder(z_dim,h2,h1,X.shape[1]).to(device)

        def model(x):
            pyro.module("decoder", dec)
            with pyro.plate("data", x.shape[0]):
                z = pyro.sample("z",
                    dist.Normal(0.,1.)
                        .expand([x.shape[0],z_dim])
                        .to_event(1)
                )
                mu,lv = dec(z)
                s = torch.exp(0.5*lv)+1e-7
                pyro.sample("obs", dist.Normal(mu,s).to_event(1), obs=x)

        def guide(x):
            pyro.module("encoder", enc)
            with pyro.plate("data", x.shape[0]):
                mu,lv = enc(x)
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

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X).to(device)),
            batch_size=batch_size, shuffle=True
        )
        for _ in range(epochs):
            for batch, in loader:
                svi.step(batch)

        # eval
        X_t = torch.from_numpy(X).to(device)
        recon = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = X_t[i:i+batch_size]
                mu,lv = enc(xb)
                s  = torch.exp(0.5*lv)+1e-7
                z  = dist.Normal(mu,s).sample()
                mu2,lv2 = dec(z)
                s2 = torch.exp(0.5*lv2)+1e-7
                err = -dist.Normal(mu2,s2).log_prob(xb)
                recon.extend(err.sum(-1).cpu().numpy())

        # threshold: 95th percentile of legit
        thr = np.percentile(np.array(recon)[y==0],95)
        y_pred = (np.array(recon)>=thr).astype(int)

        rec = recall_score(y,y_pred,pos_label=1)
        tn,fp,fn,tp = confusion_matrix(y,y_pred).ravel()
        spec = tn/(tn+fp)
        fpr,tpr,_ = roc_curve(y,np.array(recon))
        roc_auc = auc(fpr,tpr)

        # save
        pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(
            os.path.join(out_dir,f"{cust}_roc.csv"), index=False
        )
        pr, re, _ = precision_recall_curve(y,np.array(recon))
        pd.DataFrame({"precision":pr,"recall":re}).to_csv(
            os.path.join(out_dir,f"{cust}_pr.csv"), index=False
        )

        rows.append({
            "card1":      cust,
            "recall":     rec,
            "specificity":spec,
            "roc_auc":    roc_auc,
            "threshold":  thr
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir,"customer_metrics.csv"), index=False
    )
    logger.info("=== CUSTOMER TRAINING COMPLETE ===")
    return rows