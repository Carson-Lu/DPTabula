#!/usr/bin/env python3
"""
Download Credit-G dataset from OpenML and save locally.

Save path: /home/carson/projects/rrg-mijungp/carson/data/credit_g.csv
"""

import os
import pandas as pd
import openml

output_dir = "/home/carson/projects/rrg-mijungp/carson/data"
os.makedirs(output_dir, exist_ok=True) 

output_file = os.path.join(output_dir, "credit_g.csv")

# ----- 2. Download Credit-G from OpenML -----
dataset_id = 31  # Credit-G

print("Downloading Credit-G dataset from OpenML...")
dataset = openml.datasets.get_dataset(dataset_id)
X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

# ----- 3. Combine features and target -----
df = X.copy()
df[dataset.default_target_attribute] = y

# ----- 4. Save to CSV -----
df.to_csv(output_file, index=False)
print(f"Saved Credit-G dataset to {output_file}")
