#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class TabulaEmbedder:
    def __init__(self, model_path, batch_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def embed_dataframe(self, df):
        """Generate embeddings for all rows of a DataFrame"""
        embeddings = []

        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i : i + self.batch_size]

            texts = [
                ", ".join(f"{c}: {v}" for c, v in row.items())
                for _, row in batch.iterrows()
            ]

            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.inference_mode():
                emb = self.model(**inputs).last_hidden_state.mean(dim=1)

            embeddings.append(emb.cpu().numpy())

            del inputs, emb
            torch.cuda.empty_cache()

        return np.vstack(embeddings)


def compute_and_save_embeddings(csv_path, model_path, output_path, batch_size):
    print(f"Processing {csv_path}")

    df = pd.read_csv(csv_path)

    # Assume last column is target; drop it
    X = df.iloc[:, :-1]

    embedder = TabulaEmbedder(model_path, batch_size=batch_size)
    embeddings = embedder.embed_dataframe(X)

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings {embeddings.shape} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save Tabula embeddings")
    parser.add_argument("--data_dir", required=True, help="Directory with CSV splits")
    parser.add_argument("--model_path", required=True, help="Path to Tabula-8B model")
    parser.add_argument("--output_dir", required=True, help="Directory to save embeddings")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    splits = ["data_train.csv", "data_val.csv", "data_test.csv"]

    for split in splits:
        csv_file = Path(args.data_dir) / split
        out_file = Path(args.output_dir) / split.replace(".csv", "_emb.pkl")

        compute_and_save_embeddings(
            csv_file,
            args.model_path,
            out_file,
            args.batch_size,
        )
