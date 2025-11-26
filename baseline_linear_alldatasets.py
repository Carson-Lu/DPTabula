#!/usr/bin/env python3
"""
Load MULTIPLE datasets from a directory, train a logistic classifier on raw tabular data,
and evaluate performance. Writes per-dataset results and an overall summary CSV.
"""

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# -------------------------------------------------
# Parse args
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Baseline logistic regression on raw tabular data")
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CSV datasets")
parser.add_argument("--results_path", type=str, required=True, help="Path to save results CSV")
args = parser.parse_args()

data_dir = args.data_dir
results_path = args.results_path

# -------------------------------------------------
# Seed
# -------------------------------------------------
seed = 42
np.random.seed(seed)

# -------------------------------------------------
# Find all CSVs
# -------------------------------------------------
csv_files = sorted([
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.endswith(".csv")
])

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

    # Features and target
    X = df.iloc[:, :-1].copy()  # all columns except last
    y = df.iloc[:, -1].copy()   # last column as target
    y = pd.Categorical(y).codes  # convert to numeric

    # Convert categorical/text columns in X to numeric using factorize
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col], _ = pd.factorize(X[col])

    # Train/test split (stratified to ensure all classes appear)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Train classifier
    clf = LogisticRegression(max_iter=15000)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # AUC (multiclass-safe)
    try:
        y_prob = clf.predict_proba(X_test)
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

# -------------------------------------------------
# Save summary CSV
# -------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(results_path, index=False)

print("\n=== ALL DONE ===")
print(results_df)
print(f"Results written to: {results_path}")
