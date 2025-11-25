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


dataset_id = 32
dataset = openml.datasets.get_dataset(dataset_id)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

df = X.copy()
df[dataset.default_target_attribute] = y

dataset_name = dataset.name.replace(" ", "_")
output_file = os.path.join(output_dir, f"{dataset_name}.csv")

df.to_csv(output_file, index=False)
print(f"Saved {dataset_name} to {output_file}")