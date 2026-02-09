import pandas as pd
import json
import argparse
import os

# ===== Defaults =====
default_data = "BASEHOCK"
default_max_unique_ratio = 0.01
default_label_column = None  # None = use last column
base_path = "/home/carson/scratch/data_tabpe"
# ===================

def generate_metadata(csv_path, output_path, label_column=None, max_unique_ratio=0.01, hard_limit = 30):
    df = pd.read_csv(csv_path)
    
    # Default label = last column if not specified
    if label_column is None:
        label_column = df.columns[-1]

    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col == label_column:
            continue
        if df[col].dtype == "object":
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Treat numeric as categorical if few unique values
            num_unique = df[col].nunique()
            if num_unique < hard_limit or (num_unique / len(df)) < max_unique_ratio:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    metadata = {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "label": label_column
    }
    
    metadata_json = json.dumps(metadata, indent=4)
    print(metadata_json)
    
    # Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(metadata_json)
    
    return metadata

# ===== Command-line arguments =====
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=default_data, help="Dataset name (folder & file)")
parser.add_argument("--max_unique_ratio", type=float, default=default_max_unique_ratio, help="Threshold to treat numeric as categorical")
parser.add_argument("--label_column", default=default_label_column, help="Label column name (default=last column)")
args = parser.parse_args()

# Build paths from dataset name
csv_path = os.path.join(base_path, args.data, f"{args.data}.csv")
output_path = os.path.join(base_path, args.data, "metadata.json")

# Run
generate_metadata(csv_path, output_path, label_column=args.label_column, max_unique_ratio=args.max_unique_ratio)