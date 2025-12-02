#!/usr/bin/env python3
"""
Load SINGLE dataset, generate Tabula-8B embeddings in batches on GPU,
train a linear classifier, and evaluate performance. Paths are provided as arguments.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

# ----- Parse command-line arguments -----
parser = argparse.ArgumentParser(description="Tabula-8B embeddings + linear classifier on given data")
parser.add_argument("--data_path", type=str, required=True, help="Path to single CSV")
parser.add_argument("--model_path", type=str, required=True, help="Path to Tabula-8B model directory")
parser.add_argument("--results_path", type=str, required=True, help="Path to save results.txt")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for embedding generation")
args = parser.parse_args()

data_file = args.data_path
model_path = args.model_path
output_file = args.results_path
batch_size = args.batch_size

scaler = StandardScaler()

# ----- Seed for reproducibility -----
torch.manual_seed(42)
np.random.seed(42)

# ----- Load dataset -----
print("Loading dataset...")
df = pd.read_csv(data_file)
X = df.drop(columns=[df.columns[-1]])  # assume target is last column
y = pd.Categorical(df[df.columns[-1]]).codes
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ----- Train/test split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ----- Load Tabula-8B model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).to(device)
model.eval()
print("Model loaded successfully.")

# ----- Batch embedding function -----
def embed_batch(df_batch):
    """Generate embeddings for a batch of rows"""
    texts = [", ".join(f"{c}: {v}" for c, v in row.items()) for _, row in df_batch.iterrows()]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        emb = model(**inputs).last_hidden_state.mean(dim=1)
    emb_np = emb.cpu().numpy()
    del inputs, emb
    torch.cuda.empty_cache()
    return emb_np

# ----- Generate embeddings in batches -----
def generate_embeddings(X_df):
    embeddings = []
    total = len(X_df)
    checkpoints = {int(total * p / 100) for p in range(25, 101, 25)}

    for i in range(0, total, batch_size):
        batch = X_df.iloc[i:i+batch_size]
        embeddings.append(embed_batch(batch))

        current = min(i + batch_size, total)
        if current in checkpoints:
            percent = int(current / total * 100)
            print(f"{percent}% complete")

    return np.vstack(embeddings)

print("Generating embeddings for training...")
X_train_emb = generate_embeddings(X_train)
print("Generating embeddings for testing...")
X_test_emb = generate_embeddings(X_test)

# ----- Scale embeddings -----
X_train_emb = scaler.fit_transform(X_train_emb)
X_test_emb = scaler.transform(X_test_emb)

# ----- Train linear classifier -----
print("Training logistic regression...")
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train_emb, y_train)

# ----- Predictions & metrics -----
y_pred = clf.predict(X_test_emb)
acc = accuracy_score(y_test, y_pred)

try:
    y_prob = clf.predict_proba(X_test_emb)
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
except Exception:
    auc = float("nan")

dataset_name = os.path.basename(data_file)
print(f"{dataset_name}: accuracy={acc:.4f}, auc={auc:.4f}")

# ----- Save results -----
with open(output_file, "w") as f:
    f.write(f"{dataset_name}: accuracy={acc:.4f}, auc={auc:.4f}\n")

print("Results saved to:", output_file)
print("tabula_linear_singledataset.py finished successfully.")
