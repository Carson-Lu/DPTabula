#!/usr/bin/env python3
"""
Load Credit-G dataset, generate Tabula-8B embeddings, train a linear classifier,
and evaluate performance. Paths for data, model, and results are provided as arguments.
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from transformers import AutoModel, AutoTokenizer

# ----- Parse command-line arguments -----
parser = argparse.ArgumentParser(description="Tabula-8B embeddings + linear classifier on Credit-G")
parser.add_argument("--data_path", type=str, required=True, help="Path to credit_g.csv")
parser.add_argument("--model_path", type=str, required=True, help="Path to Tabula-8B model directory")
parser.add_argument("--results_path", type=str, required=True, help="Path to save results.txt")
args = parser.parse_args()

data_file = args.data_path
model_path = args.model_path
output_file = args.results_path

# ----- Load dataset -----
df = pd.read_csv(data_file)
X = df.drop(columns=[df.columns[-1]])  # default target is last column
y = df[df.columns[-1]]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----- Load Tabula-8B model -----
print("Loading Tabula-8B model from:", model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True)
model.eval()

# ----- Function to get row embeddings -----
def embed_row(row):
    """Convert a single row to Tabula-8B embedding vector."""
    text = ", ".join(f"{c}: {v}" for c, v in row.items())
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# ----- Embed datasets -----
print("Generating embeddings for training set...")
X_train_emb = [embed_row(row) for _, row in X_train.iterrows()]
print("Generating embeddings for test set...")
X_test_emb = [embed_row(row) for _, row in X_test.iterrows()]

# ----- Train linear classifier -----
print("Training logistic regression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train)

# ----- Evaluate -----
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
