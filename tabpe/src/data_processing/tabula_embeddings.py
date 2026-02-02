#!/usr/bin/env python3
import pandas as pd
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
from pathlib import Path
import os

class TabulaEmbedder:
    def __init__(self, model_path, batch_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).to(self.device)
        self.model.eval()

    def embed(self, samples, columns):
        import numpy as np
        df = pd.DataFrame(samples, columns=columns)
        embs = []
        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i:i+self.batch_size]
            texts = [", ".join(f"{c}: {v}" for c, v in row.items()) for _, row in batch.iterrows()]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.inference_mode():
                emb = self.model(**inputs).last_hidden_state.mean(dim=1)
            embs.append(emb.cpu().numpy())
            del inputs, emb
            torch.cuda.empty_cache()
        return np.vstack(embs)

def compute_and_save_embeddings(csv_path, columns, model_path, output_path, batch_size=4):
    df = pd.read_csv(csv_path)
    samples = df[columns].to_dict(orient="records")
    embedder = TabulaEmbedder(model_path, batch_size=batch_size)
    embeddings = embedder.embed(samples, columns)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings: {embeddings.shape} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with split data (train/val/test)")
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    import json
    with open(args.metadata_path) as f:
        meta = json.load(f)
    columns = meta["numerical"] + meta["categorical"]

    for split in ["data_train.csv", "data_val.csv", "data_test.csv"]:
        csv_file = Path(args.data_dir) / split
        out_file = Path(args.output_dir) / f"{split.replace('.csv', '_emb.pkl')}"
        compute_and_save_embeddings(csv_file, columns, args.model_path, out_file, args.batch_size)
