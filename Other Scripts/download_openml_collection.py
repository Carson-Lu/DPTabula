#!/usr/bin/env python3
"""
Download the most recent n datasets from an OpenML study (task collection)
and save each dataset locally as CSV.
"""

import os
import pandas as pd
import openml

# ----- Configuration -----
output_dir = "/home/carson/projects/rrg-mijungp/carson/data"
os.makedirs(output_dir, exist_ok=True)

study_id = 99  # OpenML study/collection ID
n = 10         # number of tasks to fetch

# ----- Fetch study runs -----
print(f"Fetching study {study_id}...")
study = openml.study.get_study(study_id, "runs")  # must use "runs"
all_runs = study.runs
print(f"Found {len(all_runs)} runs in study {study_id}")

# Extract unique task IDs from runs (most recent n)
all_task_ids = [run.task_id for run in all_runs]
# Remove duplicates while preserving order
seen = set()
all_task_ids = [x for x in all_task_ids if not (x in seen or seen.add(x))]

tasks_to_fetch = all_task_ids[:n]
print(f"Will fetch {len(tasks_to_fetch)} tasks: {tasks_to_fetch}")

# ----- Download datasets -----
for task_id in tasks_to_fetch:
    print(f"\nProcessing task ID: {task_id}")
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    
    dataset_name = dataset.name.replace(" ", "_")  # safe filename
    target = task.target_name
    print(f"Dataset: {dataset_name}, Target: {target}")
    
    # Download dataset
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=target)
    
    # Combine features and target
    df = X.copy()
    df[target] = y
    
    # Save CSV
    output_file = os.path.join(output_dir, f"{dataset_name}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")

print(f"\nDownloaded {len(tasks_to_fetch)} datasets successfully.")
