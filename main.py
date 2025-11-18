from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to your downloaded Tabula-8B model
model_path = "/home/carson/Scratch/hf_models/tabula-8b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# Quick test: tokenize a dummy input
sample_text = "This is a test."
inputs = tokenizer(sample_text, return_tensors="pt")

# Forward pass through the model
with torch.no_grad():
    outputs = model(**inputs)

# Print output shape to confirm it's working
print("Output logits shape:", outputs.logits.shape)

# Optional: check the first few logits
print("First few logits:", outputs.logits[0, 0, :5])
