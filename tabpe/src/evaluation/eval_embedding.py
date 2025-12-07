# ae_prdc.py
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prdc import compute_prdc
import json
import argparse
import os
import logging
import sys
from tqdm import tqdm


def make_preprocessor(cat_cols: List[str], num_cols: List[str], n_pca: Optional[int] = None) -> Pipeline:
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = Pipeline([
        ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, max(10, int(1e3))))),
        ("sc", StandardScaler())
    ])
    pre = ColumnTransformer([
        ("cat", cat, cat_cols),
        ("num", num, num_cols)
    ], remainder="drop")
    steps = [("pre", pre)]
    if n_pca is not None:
        steps.append(("pca", PCA(random_state=0)))
    return Pipeline(steps)

class AE(nn.Module):
    def __init__(self, d_in: int, d_lat: int = 32, hidden: Tuple[int, int] = (256, 128)):
        super().__init__()
        h1, h2 = hidden
        self.enc = nn.Sequential(
            nn.Linear(d_in, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, d_lat)
        )
        self.dec = nn.Sequential(
            nn.Linear(d_lat, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, d_in)
        )

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return z, xhat

@dataclass
class AEConfig:
    latent_dim: int = 32
    hidden: Tuple[int, int] = (256, 128)
    batch_size: int = 512
    lr: float = 1e-3
    max_epochs: int = 200
    patience: int = 20
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_autoencoder(X_train_np: np.ndarray, X_val_np: np.ndarray, cfg: AEConfig) -> Tuple[AE, Dict]:
    device = cfg.device
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    d_in = X_train.shape[1]
    
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = AE(d_in=d_in, d_lat=cfg.latent_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    hist = {"train_loss": [], "val_loss": []}

    for epoch in tqdm(range(cfg.max_epochs)):
        # Training phase
        model.train()
        train_losses = []
        for xb in train_loader:
            xb = xb[0].to(device)  # TensorDataset returns tuples, extract first element
            z, xhat = model(xb)
            loss = crit(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb in val_loader:
                xb = xb[0].to(device)  # TensorDataset returns tuples, extract first element
                z, xhat = model(xb)
                loss = crit(xhat, xb)
                val_losses.append(loss.item())
        
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)

        # Early stopping on validation loss (not training loss)
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_loss": best_loss, **hist}

@torch.no_grad()
def embed_with_ae(model: AE, X_np: np.ndarray, cfg: AEConfig) -> np.ndarray:
    device = cfg.device
    model = model.to(device)  # Ensure model is on correct device
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    zs = []
    bs = 8192
    for i in range(0, X.shape[0], bs):
        z, _ = model(X[i:i+bs])
        zs.append(z.detach().cpu().numpy())
    return np.concatenate(zs, axis=0)

@dataclass
class EmbedConfig:
    n_pca: Optional[int] = None  

def fit_embedder_on_real(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    embed_cfg: EmbedConfig = EmbedConfig(),
    ae_cfg: AEConfig = AEConfig(),
) -> Tuple[Pipeline, AE, Dict]:
    pipe = make_preprocessor(cat_cols, num_cols, n_pca=embed_cfg.n_pca)
    Xr_proc = pipe.fit_transform(X_train)  # fit on TRAIN ONLY    
    X_val_proc = pipe.transform(X_val).astype(np.float32)
    
    model, logs = train_autoencoder(Xr_proc.astype(np.float32), X_val_proc.astype(np.float32), ae_cfg)
    return pipe, model, logs

def embed_real_and_synth(
    preproc: Pipeline,
    ae: AE,
    X_real: pd.DataFrame,
    X_synth: pd.DataFrame,
    ae_cfg: AEConfig = AEConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    Xr_proc = preproc.transform(X_real).astype(np.float32)
    Xf_proc = preproc.transform(X_synth).astype(np.float32)
    Z_real = embed_with_ae(ae, Xr_proc, ae_cfg)
    Z_fake = embed_with_ae(ae, Xf_proc, ae_cfg)
    return Z_real, Z_fake

def evaluate_downstream_task(
    Z_real_train: np.ndarray, 
    y_real_train: np.ndarray,
    Z_real_test: np.ndarray,
    y_real_test: np.ndarray,
    random_state: int = 42
) -> Dict:
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_real_train, y_real_test])
    label_encoder.fit(all_labels)
    y_real_train = label_encoder.transform(y_real_train)
    y_real_test = label_encoder.transform(y_real_test)
    
    
    # Train classifier on real data embeddings
    clf = XGBClassifier(random_state=random_state)
    clf.fit(Z_real_train, y_real_train)
    
    # Evaluate on real test data
    y_real_pred = clf.predict(Z_real_test)
    accuracy = accuracy_score(y_real_test, y_real_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': classification_report(y_real_test, y_real_pred, output_dict=True)
    }

def compute_prdc_(Z_real, Z_fake, nearest_k) -> float:
    Z_real_ = Z_real.astype(np.float32)
    Z_fake_ = Z_fake.astype(np.float32)
    if len(Z_real) > 20000:
        np.random.seed(0)
        indices = np.random.choice(len(Z_real), 20000, replace=False)
        Z_real_ = Z_real_[indices]
    
    if len(Z_fake) > 20000:
        np.random.seed(0)
        indices = np.random.choice(len(Z_fake), 20000, replace=False)
        Z_fake_ = Z_fake_[indices]
    
    prdc = compute_prdc(Z_real_, Z_fake_, nearest_k)
    return prdc

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = args.dataset
    seed = args.seed
    synthetic_dir = args.synthetic_dir


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        filename=f"{synthetic_dir}/eval_prdc.log",
        filemode='w'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'))
    logging.getLogger().addHandler(console)


    df_real_train = pd.read_csv(f"data/processed/{dataset}/{seed}/data_train.csv")
    df_real_val = pd.read_csv(f"data/processed/{dataset}/{seed}/data_val.csv")
    df_real_test = pd.read_csv(f"data/processed/{dataset}/{seed}/data_test.csv")

    metadata = json.load(open(f"data/{dataset}/metadata.json"))
    target_col = metadata["label"]
    
    y_real_train = df_real_train[target_col]
    y_real_test = df_real_test[target_col]

    df_real_train = df_real_train.drop(columns=[target_col])
    df_real_val = df_real_val.drop(columns=[target_col])
    df_real_test = df_real_test.drop(columns=[target_col])
    ae_cfg = AEConfig(latent_dim=32, hidden=(256, 128), batch_size=256, max_epochs=100, patience=5)
    preproc, ae, logs = fit_embedder_on_real(
        df_real_train,
        df_real_val,
        cat_cols=metadata["categorical"],
        num_cols=metadata["numerical"],
        embed_cfg=EmbedConfig(n_pca=None),  
        ae_cfg=ae_cfg
    )
    logging.info(f"AE best val reconstruction loss: {logs['best_val_loss']:.6f}")

    # Transform test data
    X_real_train_proc = preproc.transform(df_real_train).astype(np.float32)
    X_real_test_proc = preproc.transform(df_real_test).astype(np.float32)

    # Compute test reconstruction loss
    ae.eval()
    with torch.no_grad():
        device = ae_cfg.device  # Use consistent device from config
        ae = ae.to(device)  # Ensure model is on correct device
        X_real_test_tensor = torch.tensor(X_real_test_proc, dtype=torch.float32).to(device)
        z_real_test, xhat_real_test = ae(X_real_test_tensor)
        test_loss = nn.MSELoss()(xhat_real_test, X_real_test_tensor).item()
    
    logging.info(f"Test reconstruction loss: {test_loss:.6f}")
    
    # ---- Embed test data and evaluate downstream task ----
    Z_real_train = embed_with_ae(ae, X_real_train_proc, ae_cfg)
    Z_real_test = embed_with_ae(ae, X_real_test_proc, ae_cfg)    
    # Evaluate downstream task on test set
    test_results = evaluate_downstream_task(Z_real_train, y_real_train, Z_real_test, y_real_test)  # No split needed for test
    logging.info(f"Test accuracy: {test_results['accuracy']:.4f}")
    logging.info(f"Test macro avg F1: {test_results['classification_report']['macro avg']['f1-score']:.4f}")

    prdc_upper_bound = compute_prdc_(Z_real_train, Z_real_test, 5)
    logging.info(f"PRDC upper bound: {prdc_upper_bound}")



    # check list of dirs in synthetic_dir
    epochs = os.listdir(synthetic_dir)
    epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]
    epochs.sort()
    for epoch in epochs:
        if not os.path.isdir(f"{synthetic_dir}/{epoch}"):
            continue
        
        logging.info(f"Epoch {epoch}:")
        df_fake = pd.read_csv(f"{synthetic_dir}/{epoch}/synthetic_df.csv")
        y_fake = df_fake[target_col]
        df_fake = df_fake.drop(columns=[target_col])
        X_fake_proc = preproc.transform(df_fake).astype(np.float32)
        Z_fake = embed_with_ae(ae, X_fake_proc, ae_cfg)
        
        synthetic_results = evaluate_downstream_task(Z_fake, y_fake, Z_real_test, y_real_test)
        logging.info(f"\tTest accuracy: {synthetic_results['accuracy']:.4f}")
        prdc = compute_prdc_(Z_real_train, Z_fake, 5)
        logging.info(f"\tPRDC: {prdc}")
        
        mu1, sigma1 = Z_real_train.mean(axis=0), np.cov(Z_real_train, rowvar=False)
        mu2, sigma2 = Z_fake.mean(axis=0), np.cov(Z_fake, rowvar=False)        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--synthetic_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)