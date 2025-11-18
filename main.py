from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/carson/scratch/hf_models/tabula-8b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Load model with automatic device mapping (offloads layers to CPU/GPU as needed)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",     # Automatically splits model across CPU/GPU
    dtype=torch.float16  # Use half-precision to save memory
)

# Tokenize
sample_text = "Hello world"
inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

print("Output logits shape:", outputs.logits.shape)
