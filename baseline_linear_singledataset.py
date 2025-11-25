#!/usr/bin/env python3
"""
Baseline: Train a simple logistic regression on raw tabular data from CSV
and evaluate performance.

Text columns are automatically encoded using pandas factorize.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ----- Parse command-line arguments -----
parser = argparse.ArgumentParser(description="Baseline logistic regression on raw tabular data")
parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
parser.add_argument("--results_path", type=str, required=True, help="Path to save results.txt")
args = parser.parse_args()

data_file = args.data_path
output_file = args.results_path

print(f"Loading dataset from: {data_file}")
df = pd.read_csv(data_file)

# ----- Preprocess -----
X = df.drop(columns=[df.columns[-1]])  # last column as target
y = df[df.columns[-1]]

# Convert categorical/text columns to numeric using factorize
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col], _ = pd.factorize(X[col])

# ----- Split train/test -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ----- Train logistic regression -----
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)
print("Training completed.")

# ----- Evaluate -----
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

if y_prob is not None:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.4f}")

# ----- Save results -----
with open(output_file, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    if y_prob is not None:
        f.write(f"AUC: {auc:.4f}\n")

print(f"Results saved to: {output_file}")
