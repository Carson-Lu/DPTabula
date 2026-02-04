import scipy.io
import pandas as pd
import os
import argparse

# ===== Defaults =====
default_data = "BASEHOCK"
base_path = "/home/carson/scratch/data_tabpe"

# ===== Argument parser =====
parser = argparse.ArgumentParser(description="Convert .mat dataset to CSV")
parser.add_argument("--data", default=default_data, help="Dataset name (folder & file base name)")
args = parser.parse_args()

data_name = args.data

# Build paths dynamically
mat_file = os.path.join(base_path, data_name, f"{data_name}.mat")
csv_file = os.path.join(base_path, data_name, f"{data_name}.csv")
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# Load .mat file
data = scipy.io.loadmat(mat_file)
data = {k: v for k, v in data.items() if not k.startswith('__') and 'readme' not in k}

# Extract features
X = data.get('X')
if X is None:
    raise ValueError("No 'X' found in .mat file")

# Convert sparse matrix to dense if needed
if hasattr(X, "todense"):
    X = X.todense()
elif hasattr(X, "toarray"):
    X = X.toarray()

X = pd.DataFrame(X)

# Extract labels
y = data.get('Y')
if y is None:
    raise ValueError("No 'Y' found in .mat file")

y = pd.DataFrame(y, columns=['label'])

# Combine and save
df = pd.concat([X, y], axis=1)
df.to_csv(csv_file, index=False)
print(f"Saved CSV: {csv_file} (shape={df.shape})")
