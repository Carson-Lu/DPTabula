#!/usr/bin/env python3
"""
Download multiple datasets from OpenML using dataset IDs and save locally.
"""

import os
import pandas as pd
import openml

# ----- Configuration -----
output_dir = "/home/carson/projects/rrg-mijungp/carson/data"
os.makedirs(output_dir, exist_ok=True)

# Use project-local cache to avoid ~/.cache issues
openml.config.cache_directory = "/home/carson/projects/rrg-mijungp/carson/openml_cache"
os.makedirs(openml.config.cache_directory, exist_ok=True)

# Dataset IDs to download
dataset_ids = [1464, 1063, 3, 1067, 1494, 1510, 1068, 1050, 37, 1480, 1489, 1049]

for dataset_id in dataset_ids:
    print(f"\nProcessing dataset ID {dataset_id}...")
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        df = X.copy()
        df[dataset.default_target_attribute] = y

        dataset_name = dataset.name.replace(" ", "_")
        output_file = os.path.join(output_dir, f"{dataset_name}.csv")

        df.to_csv(output_file, index=False)
        print(f"Saved {dataset_name} to {output_file}")
        
    except Exception as e:
        print(f"Error fetching dataset {dataset_id}: {e}")

print("\nAll datasets processed.")
