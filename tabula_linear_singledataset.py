#!/usr/bin/env python3
"""
Load SINGLE dataset, generate Tabula-8B embeddings on GPU if available,
train a linear classifier, and evaluate performance. Paths are provided as arguments.
"""

import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

# ----- Parse command-line arguments -----
parser = argparse.ArgumentParser(description="Tabula-8B embeddings + linear classifier on given data")
parser.add_argument("--data_path", type=str, required=True, help="Path to single CSV")
parser.add_argument("--model_path", type=str, required=True, help="Path to Tabula-8B model directory")
parser.add_argument("--results_path", type=str, required=True, help="Path to save results.txt")
args = parser.parse_args()

data_file = args.data_path
model_path = args.model_path
output_file = args.results_path


# ----- Seed for reproducibility -----
torch.manual_seed(42)
np.random.seed(42)

# ----- Load dataset -----
print("Loading dataset...")
df = pd.read_csv(data_file)
X = df.drop(columns=[df.columns[-1]])  # assume target is last column
y = df[df.columns[-1]]
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Split train/test
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ----- Load Tabula-8B model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("Loading Tabula-8B model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    raise

model = model.to(device)
model.eval()
print("Model loaded successfully.")

# ----- Function to get row embeddings -----
def embed_row(row):
    """Convert a single row to Tabula-8B embedding vector on GPU."""
    text = ", ".join(f"{c}: {v}" for c, v in row.items())
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()  # move to CPU for sklearn

# ----- Embed datasets -----
print("Generating embeddings for training...")
X_train_emb = [embed_row(row) for _, row in X_train.iterrows()]
print("Generating embeddings for testing...")
X_test_emb = [embed_row(row) for _, row in X_test.iterrows()]

# ----- Train linear classifier -----
print("Training logistic regression...")
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_emb, y_train)

# ----- Evaluate -----
print("Evaluating model...")
y_pred = clf.predict(X_test_emb)
y_prob = clf.predict_proba(X_test_emb)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")

# ----- Save results -----
with open(output_file, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nAUC: {auc:.4f}\n")

print("Results saved to:", output_file)
print("tabula_linear_test.py finished successfully.")
