#!/usr/bin/env python3
"""
Load MULTIPLE datasets from a directory, generate Tabula-8B embeddings
(train/test) in batches, train a linear classifier, and evaluate performance.

Writes per-dataset results and an overall summary.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

# -------------------------------------------------
# Parse args
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Tabula-8B embedding classifier for multiple datasets")
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing multiple CSVs")
parser.add_argument("--model_path", type=str, required=True, help="Path to Tabula-8B model")
parser.add_argument("--results_path", type=str, required=True, help="Path to write results output CSV")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation")
args = parser.parse_args()

data_dir = args.data_dir
model_path = args.model_path
results_path = args.results_path
batch_size = args.batch_size

scaler = StandardScaler()

# -------------------------------------------------
# Seed
# -------------------------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# -------------------------------------------------
# Load Tabula model + tokenizer
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).to(device)
model.eval()

# -------------------------------------------------
# Embedding function (batch)
# -------------------------------------------------
def embed_batch(df_batch):
    """Convert a dataframe batch into embeddings"""
    texts = [", ".join(f"{c}: {v}" for c, v in row.items()) for _, row in df_batch.iterrows()]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1)
    return emb.cpu().numpy()

# -------------------------------------------------
# Find all CSVs
# -------------------------------------------------
csv_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")])
if not csv_files:
    raise RuntimeError("No CSV files found")
print(f"Found {len(csv_files)} datasets.")

# -------------------------------------------------
# Process each dataset
# -------------------------------------------------
results = []

for csv_file in csv_files:
    dataset_name = os.path.basename(csv_file)
    print(f"\n=== Processing {dataset_name} ===")

    df = pd.read_csv(csv_file)

    X = df.drop(columns=[df.columns[-1]])
    y = pd.Categorical(df[df.columns[-1]]).codes

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Embeddings in batches
    X_train_emb = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train.iloc[i:i+batch_size]
        X_train_emb.append(embed_batch(batch))
    X_train_emb = np.vstack(X_train_emb)

    X_test_emb = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test.iloc[i:i+batch_size]
        X_test_emb.append(embed_batch(batch))
    X_test_emb = np.vstack(X_test_emb)

    # Scale embeddings
    X_train_emb = scaler.fit_transform(X_train_emb)
    X_test_emb = scaler.transform(X_test_emb)

    # Train classifier
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train_emb, y_train)

    # Predictions
    y_pred = clf.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)

    # AUC (only if probabilities available)
    try:
        y_prob = clf.predict_proba(X_test_emb)
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except Exception:
        auc = float("nan")

    print(f"{dataset_name}: accuracy={acc:.4f}, auc={auc:.4f}")

    results.append({
        "dataset": dataset_name,
        "accuracy": acc,
        "auc": auc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(np.unique(y)),
    })

    # Free GPU memory between datasets
    torch.cuda.empty_cache()

# -------------------------------------------------
# Save summary CSV
# -------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(results_path, index=False)

print("\n=== ALL DONE ===")
print(results_df)
print(f"Results written to: {results_path}")
