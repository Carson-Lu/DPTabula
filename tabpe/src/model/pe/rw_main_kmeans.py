import os
import csv
import copy
import math
import json
import scipy
import torch
import random
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def find_required_noise_multiplier(epsilon, num_steps, num_N):
    delta= 1/(num_N*math.log(num_N))
    def delta_Gaussian(eps, mu):
        """Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu"""
        if mu==0:
            return 0
        return scipy.stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu - mu / 2)
    def eps_Gaussian(delta, mu):
        """Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu"""
        def f(x):
            return delta_Gaussian(x, mu) - delta
        return scipy.optimize.root_scalar(f, bracket=[0, 500], method='brentq').root
    def compute_epsilon(noise_multiplier, num_steps, delta):
        return eps_Gaussian(delta, np.sqrt(num_steps) / noise_multiplier)
    def objective(x):
        return -compute_epsilon(x[0], num_steps, delta)
    def constraints(x):
        return (epsilon - .00001) - compute_epsilon(x[0], num_steps, delta)

    output = scipy.optimize.minimize(lambda x: objective(x), x0=[1], bounds=[(0, None)], constraints={'type': 'ineq', 'fun': constraints})
    assert(output.success)
    assert(-output.fun < epsilon)
    return output.x[0]

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

# MODIFIED =========================================================
def tabpe_get_embeddings(samples, columns, info):
    return np.array([get_vector(s, columns, info) for s in samples])

def vote(public, private, count, noise_multiplier, embed_fn):
    public_embeddings = embed_fn(public)
    private_embeddings = embed_fn(private)

    distances = torch.cdist(
        torch.tensor(private_embeddings),
        torch.tensor(public_embeddings)
    ).cpu().numpy()

    votes_embeddings = distances.argmin(axis=1).tolist()
    histogram = [0 for _ in range(len(public))]
    for v in votes_embeddings:
        histogram[v] += 1
    noisy_histogram = [h + np.random.normal(0, noise_multiplier) for h in histogram]
    public_best = []
    for i in np.argsort(noisy_histogram)[::-1][:count]:
        public_best.append(public[i])
    return public_best, histogram, noisy_histogram

# END OF MODIFIED ==================================================

def try_float(x):
    try:
        return float(x)
    except Exception as e:
        return 0

def parse_args():
    parser = argparse.ArgumentParser(description='PE for tabular data')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--sampling_epochs', default=0, type=int, help='Number of epochs for PE1 with sampling')
    parser.add_argument('--priv_train_csv', default=None, type=str, help='Location of private data')
    parser.add_argument('--metadata_path', default=None, type=str, help='Path to metadata json file')
    parser.add_argument('--num_samples', default=2000, type=int, help='Number of samples to generate')
    parser.add_argument('--num_variations', default=3, type=int, help='How many variations for every sample')
    parser.add_argument('--variance_multiplier', default=0.5, type=float, help='Multiplier for variance')
    parser.add_argument('--decay_type', default='linear', type=str, help='Type of decay for variance')
    parser.add_argument('--gamma', default=0.2, type=float, help='Gamma for decay')
    parser.add_argument('--output_dir', default=None, type=str, help='Output directory')
    parser.add_argument('--epsilon', default=-1, type=float, help='Privacy epsilon value')

    # ADDED ==============================================================
    parser.add_argument('--model_path', type=str, required=True, help='Path to embedding model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for embedding generation')
    parser.add_argument('--generator_method', type=str, default='tabula', help='Which generator to use')
    parser.add_argument('--compare_method', type=str, default='tabula', help='Which comparison method to use')
    # END OF ADDED ========================================================
    args = parser.parse_args()

    return args

class LLMEmbedder:
    def __init__(self, model_path, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

    def embed(self, samples, columns):
        df = pd.DataFrame(samples, columns=columns)
        embeddings = []

        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i:i+self.batch_size]
            texts = [", ".join(f"{c}: {v}" for c, v in row.items()) for _, row in batch.iterrows()]

            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.inference_mode():
                emb = self.model(**inputs).last_hidden_state.mean(dim=1)

            embeddings.append(emb.cpu().numpy())
            del inputs, emb
            torch.cuda.empty_cache()

        return np.vstack(embeddings)

def main(args):
    # ADDED ==============================================================
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # END OF ADDED ========================================================

    with open(args.metadata_path, 'r') as f:
        columns = json.load(f)

    assert (args.priv_train_csv is not None), "priv_train_csv must be provided"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f)
    
    priv_data_path = args.priv_train_csv
    assert priv_data_path.endswith(".csv"), "data_train must be a csv file"

    private = defaultdict(list)
    with open(priv_data_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in rows:
            label = row[columns["label"]]
            values = [try_float(row[c]) for c in columns["numerical"]] + [row[c] for c in columns["categorical"]]
            private[label].append(values)

    info = {}
    for c in columns["numerical"]:
        info[c] = {'min': float('inf'), 'max': float('-inf')}
        for r in rows:
            v = try_float(r[c])
            if v < info[c]['min']:
                info[c]['min'] = v
            if v > info[c]['max']:
                info[c]['max'] = v
    for c in columns["categorical"]:
        info[c] = {}
        for r in rows:
            if r[c] not in info[c]:
                info[c][r[c]] = len(info[c])
    logging.info(info)

    total = sum([len(private[k]) for k in private])
    num_samples = args.num_samples if args.num_samples > 0 else total
    counts = {label: math.floor(num_samples * (len(private[label]) / total)) for label in private}
    counts[ list(counts.keys())[0] ] += num_samples - sum(counts.values())

    noise_multiplier = 0.0 if args.epsilon == -1 else find_required_noise_multiplier(args.epsilon, args.epochs, total)
    logging.info(f'Noise multiplier: {noise_multiplier} for epsilon {args.epsilon}')

    for e in range(2 + args.epochs):
        if e == 1 + args.epochs:
            break
        if os.path.exists(f'{args.output_dir}/{e}/synthetic_df.csv'):
            logging.info(f'File {args.output_dir}/{e}/synthetic_df.csv exists.')
        else:
            logging.info(f'File {args.output_dir}/{e}/synthetic_df.csv does not exist.')
            break

    if e == 0:
        def random_sample():
            numerical = [random.uniform(info[c]['min'], info[c]['max']) for c in columns["numerical"]]
            categorical = [random.choice(list(info[c].keys())) for c in columns["categorical"]]
            return numerical + categorical

        public = defaultdict(list)
        for label in counts:
            for _ in range(counts[label]):
                public[label].append(random_sample())
        
        os.makedirs(f'{args.output_dir}/0', exist_ok=True)
        with open(f'{args.output_dir}/0/synthetic_df.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=columns["numerical"] + columns["categorical"] + [columns["label"]])
            writer.writeheader()
            for label in public:
                for sample in public[label]:
                    row = {c: sample[i] for i, c in enumerate(columns["numerical"] + columns["categorical"])}
                    row[columns["label"]] = label
                    writer.writerow(row)
    else:
        with open(f'{args.output_dir}/{e - 1}/synthetic_df.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            public = defaultdict(list)
            for row in rows:
                label = row[columns["label"]]
                values = [float(row[c]) for c in columns["numerical"]] + [row[c] for c in columns["categorical"]]
                public[label].append(values)
    
    # ADDED ===============================================================
    all_columns = columns["numerical"] + columns["categorical"]

    if args.compare_method == "tabpe":
        logging.info("Using TabPE embedding backend")

        def embed_fn(samples):
            return tabpe_get_embeddings(samples, columns, info)

    else:
        logging.info("Using LLM embedding backend")

        embedder = LLMEmbedder(args.model_path, args.batch_size)

        def embed_fn(samples):
            return embedder.embed(samples, all_columns)
    # END OF ADDED ========================================================

    histograms = {}
    for epoch in tqdm(range(max(e, 1), 1 + args.epochs), desc="Epochs"):
        def linear_range(epoch_, T, base, floor=0.02):
            t = epoch_ / T
            return base - (base - floor) * t
        
        def var_range(epoch_, T, base, floor=0.02):
            t = epoch_ / T
            return base - (base - floor) * (t**args.gamma)  # quadratic decay

        def auto_range(epoch_, T, base, floor=0.02):
            if args.decay_type == 'linear':
                return linear_range(epoch_, T, base, floor)
            elif args.decay_type == 'polynomial':
                return var_range(epoch_, T, base, floor)
            else:
                raise ValueError(f'Invalid decay type: {args.decay_type}')

        # logging.info(f'Epoch {epoch}')
        def variation_sample(sample, epoch_=epoch):
            s = copy.deepcopy(sample)
            for i, c in enumerate(columns["numerical"] + columns["categorical"]):
                if c in columns["numerical"]:
                    # Always add noise first
                    multiplier_range = auto_range(epoch_, args.epochs, args.variance_multiplier)
                    s[i] += random.uniform(-multiplier_range, multiplier_range) * (info[c]['max'] - info[c]['min'])
                    
                    if s[i] < info[c]['min']:
                        s[i] = info[c]['min']
                    if s[i] > info[c]['max']:
                        s[i] = info[c]['max']
                else:
                    probability = auto_range(epoch_, args.epochs, args.variance_multiplier)
                    if np.random.choice([0, 1], 1, p=[1.0 - probability, probability])[0] == 1:
                        s[i] = random.choice(list(info[c].keys()))
            return s

        public_new = defaultdict(list)
        histograms[epoch] = {}
        for label in public:
            # Get vote counts for this label's samples
            label_public = public[label]
            label_private = private[label]
            private_embeddings = embed_fn(label_private)
            
            if epoch <= args.sampling_epochs:
                # PE1 with sampling
                # denoise vote counts
                # Calculate distances and get vote counts

                label_embeddings = embed_fn(label_public)
                distances = torch.cdist(torch.Tensor(private_embeddings), torch.Tensor(label_embeddings)).cpu().numpy()
                votes_embeddings = distances.argmin(axis=1).tolist()
                
                # Count votes for each public sample
                clean_vote_counts = [0 for _ in range(len(label_public))]
                for v in votes_embeddings:
                    clean_vote_counts[v] += 1
                
                # Add noise to vote counts if dp is enabled
                if noise_multiplier > 0:
                    vote_counts = [h + np.random.normal(0, noise_multiplier) for h in clean_vote_counts]
                else:
                    vote_counts = clean_vote_counts

                vote_counts = [h if h > 0 else 0 for h in vote_counts]

                # Calculate probabilities based on vote counts
                total_votes = sum(vote_counts)
                probabilities = [count / total_votes for count in vote_counts]
                # Sample with replacement based on vote probabilities
                samples_to_generate = len(label_public)
                sampled_indices = np.random.choice(
                    len(label_public),
                    size=samples_to_generate, 
                    replace=True, 
                    p=probabilities
                )
                    
                # Generate variations for sampled data
                for idx in sampled_indices:
                    variation = variation_sample(label_public[idx])
                    public_new[label].append(variation)
                
                # Store histograms
                histograms[epoch][label] = {
                    'clean': clean_vote_counts,
                    'noisy': vote_counts
                }
            else:
                # PE with ranking selection
                # Generate variations for all samples
                public_with_variations = copy.deepcopy(label_public)
                for public_sample in label_public:
                    for _ in range(args.num_variations):
                        public_with_variations.append(variation_sample(public_sample))
                
                # Then use vote function to select best samples
                public_new[label], clean_histogram, noisy_histogram = vote(
                    public_with_variations, 
                    label_private, 
                    len(label_public),
                    noise_multiplier,
                    embed_fn # ADDED
                )
                
                # Store histograms
                histograms[epoch][label] = {
                    'clean': clean_histogram,
                    'noisy': noisy_histogram
                }

        os.makedirs(f'{args.output_dir}/{epoch}', exist_ok=True)
        with open(f'{args.output_dir}/{epoch}/synthetic_df.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=columns["numerical"] + columns["categorical"] + [columns["label"]])
            writer.writeheader()
            for label in public_new:
                for sample in public_new[label]:
                    row = {c: sample[i] for i, c in enumerate(columns["numerical"] + columns["categorical"])}
                    row[columns["label"]] = label
                    writer.writerow(row)
        
        public = copy.deepcopy(public_new)

    with open(f'{args.output_dir}/histograms.pkl', 'wb') as f:
        pkl.dump(histograms, f)

    logging.info(f'PE finished')

    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)



def dp_resample_synthetic(
    syn_data,
    real_data,
    num_synthetic_samples,
    num_clusters=None, # cluster_selection == ch_index will ignore this
    cluster_selection="fixed", # Fixed or classes
    max_K=1000,
    epsilon=1.0,
    delta=1e-5
):
    if cluster_selection == "classes": # LINE 1
        if num_clusters is None:
            raise ValueError("n_classes must be provided for cluster_selection='classes'")
        K = num_clusters
    elif cluster_selection == "ch_index":
        best_score = -np.inf
        best_K = 2
        for k in range(2, min(max_K, len(syn_data)) + 1):
            labels = KMeans(n_clusters=k, random_state=0).fit_predict(syn_data)
            score = calinski_harabasz_score(syn_data, labels)
            if score > best_score:
                best_score = score
                best_K = k
        K = best_K
    else:
        K = num_clusters if num_clusters else 10

    # K-means on synthetic embeddings
    kmeans = KMeans(n_clusters=K, random_state=0).fit(syn_data)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # LINE 3-8
    h = np.zeros(K)
    for e in real_data:
        h[np.argmin(np.linalg.norm(e - centroids, axis=1))] += 1

    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noisy_h = np.maximum(h + np.random.normal(0, sigma, size=K), 0)
    p = noisy_h / noisy_h.sum()

    # LINE 9-14
    sampled_indices = []
    for k in range(K):
        cluster_idx = np.where(clusters == k)[0]
        n_samples = int(np.floor(num_synthetic_samples * p[k]))
        if n_samples > 0 and len(cluster_idx) > 0:
            replace = n_samples > len(cluster_idx)
            sampled_indices.extend(np.random.choice(cluster_idx, n_samples, replace=replace))

    return p, sampled_indices