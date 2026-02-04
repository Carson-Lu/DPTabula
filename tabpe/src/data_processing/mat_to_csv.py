import scipy.io
import pandas as pd
import os

# Paths
mat_file = '/home/carson/scratch/data_tabpe/BASEHOCK/BASEHOCK.mat'
csv_file = '/home/carson/scratch/data_tabpe/BASEHOCK/BASEHOCK.csv'
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
    X = X.todense()  # now X is a regular numpy matrix
elif hasattr(X, "toarray"):
    X = X.toarray()

X = pd.DataFrame(X)

# Extract labels
y = data.get('Y')
if y is None:
    raise ValueError("No 'Y' found in .mat file")

y = pd.DataFrame(y, columns=['label'])

# Combine
df = pd.concat([X, y], axis=1)

# Save CSV
df.to_csv(csv_file, index=False)
print(f"Saved CSV: {csv_file} (shape={df.shape})")
