#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import save_file

class TabulaEmbedder:
    def __init__(self, model_path, batch_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True).to(self.device)
        self.model.eval()

    def embed_dataframe(self, df, output_dir=None, save_every_percent=10):
        embeddings = []
        total = len(df)
        checkpoint_dir = output_dir / "checkpoints" if output_dir else None
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Number of rows per checkpoint
        step = max(1, (total * save_every_percent + 99) // 100)

        for i in range(0, total, self.batch_size):
            batch = df.iloc[i : i + self.batch_size]
            texts = [", ".join(f"{c}: {v}" for c, v in row.items()) for _, row in batch.iterrows()]

            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.inference_mode():
                emb = self.model(**inputs).last_hidden_state.mean(dim=1)

            embeddings.append(emb.cpu())
            del inputs, emb

            # Progress logging and checkpoint
            if (i + self.batch_size) % step < self.batch_size:
                pct = min(100, (i + self.batch_size) * 100 // total)
                print(f"Progress: {pct}% ({i + self.batch_size}/{total})", flush=True)
                if checkpoint_dir:
                    save_file({"embeddings": torch.cat(embeddings, dim=0)}, checkpoint_dir / f"checkpoint_{pct}.safetensors")

        return torch.cat(embeddings, dim=0)


def compute_and_save_embeddings(csv_path, metadata_path, model_path, output_dir, batch_size):
    # Load metadata to identify label column
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    label_col = metadata["label"]

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols]

    embedder = TabulaEmbedder(model_path, batch_size=batch_size)
    embeddings = embedder.embed_dataframe(X, output_dir=Path(output_dir), save_every_percent=10)

    # Final save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "data_train_emb.safetensors"
    save_file({"embeddings": embeddings}, out_file)
    print(f"Saved final embeddings {tuple(embeddings.shape)} → {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    compute_and_save_embeddings(
        csv_path=args.data_path,
        metadata_path=args.metadata_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
