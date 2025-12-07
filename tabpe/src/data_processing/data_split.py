import random
import os
from pathlib import Path
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def data_split(file_path, outdir, seed):
    """    
    Process the data and save it in the output directory.
    Args:
        file_path: Path to the data file.
        outdir: Path to the output directory.
        seed: Random seed.
    Returns:
        None

    Example:
        python data_split.py --file_path data/adult/adult_all_csv.csv --outdir data/processed/adult --seed 42
        The output will be in data/processed/adult/42/data_train.csv, data/processed/adult/42/data_val.csv, data/processed/adult/42/data_test.csv
    """

    logging.info(f"Processing {file_path} with output directory {outdir} and seed {seed}")
    file_path = Path(file_path)
    metadata_path = file_path.parent / "metadata.json"
    # create output directory if it does not exist with Path 
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # check if folder is empty
    dirs_ = os.listdir(outdir_path)
    if str(seed) in dirs_:
        seed_dir = outdir_path / str(seed)
        if os.path.exists(seed_dir / "data_train.csv") and os.path.exists(seed_dir / "data_val.csv") and os.path.exists(seed_dir / "data_test.csv"):
            logging.info(f"File {file_path} has been already processed with seed {seed}. Skipping processing.")
            return
    else:
        # create folder with seed
        seed_dir = outdir_path / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    all_columns = metadata["numerical"] + metadata["categorical"] + [metadata["label"]]

    df = pd.read_csv(file_path)
    org_num_rows = len(df)
    logging.info(f"Data loaded with {org_num_rows} rows and {len(all_columns)} columns.")

    # remove rows with missing values or nan
    df = df.dropna()
    num_rows_removed = org_num_rows - len(df)
    logging.info(f"Removed {num_rows_removed} rows with missing values or nan.")
    
    df_train, df_val_test = train_test_split(df, test_size=0.3, random_state=seed, stratify=df[metadata["label"]])
    df_val, df_test = train_test_split(df_val_test, test_size=0.5, random_state=seed, stratify=df_val_test[metadata["label"]])
    
    df_train.to_csv(seed_dir / "data_train.csv", index=False)
    df_val.to_csv(seed_dir / "data_val.csv", index=False)
    df_test.to_csv(seed_dir / "data_test.csv", index=False)
    
    logging.info(f"Data processed and saved in {seed_dir}")
