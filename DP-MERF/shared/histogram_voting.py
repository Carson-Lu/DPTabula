import torch
import numpy as np

def get_vector(s, columns, info):
    vector = []

    for i, c in enumerate(columns["numerical"] + columns["categorical"]):
        if c in columns["numerical"]:
            vector.append((s[i] - info[c]['min']) / (info[c]['max'] - info[c]['min']))
        else:
            vector_add = [0.0] * len(info[c])
            vector_add[info[c][s[i]]] = 1.0 / 3.0
            vector += vector_add
    return vector

def tabpe_get_embeddings(samples, columns, info):
    return np.array([get_vector(s, columns, info) for s in samples], dtype=np.float32)

def vote(public, public_embeddings, private_embeddings, count, noise_multiplier, chunk_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    public_t = torch.from_numpy(public_embeddings).to(device=device, dtype=torch.float32)

    num_public = len(public_embeddings)
    histogram = np.zeros(num_public, dtype=np.int64)

    for i in range(0, len(private_embeddings), chunk_size):
        private_chunk = torch.from_numpy(
            private_embeddings[i:i+chunk_size]
        ).to(device=device, dtype=torch.float32)

        distances = torch.cdist(private_chunk, public_t)
        votes = distances.argmin(dim=1).cpu().numpy()

        np.add.at(histogram, votes, 1)

        del distances, private_chunk

    noisy_histogram = histogram + np.random.normal(0, noise_multiplier, size=num_public)
    public_best = []
    for i in np.argsort(noisy_histogram)[::-1][:count]:
        public_best.append(public[i])
        
    del public_t
    torch.cuda.empty_cache()
        
    return public_best, histogram, noisy_histogram