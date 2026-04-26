import copy
import random
import torch
import numpy as np
import math
import scipy
from collections import defaultdict


def linear_range(epoch_, T, base, floor=0.02):
    t = epoch_ / T
    return base - (base - floor) * t


def polynomial_range(epoch_, T, base, floor=0.02, gamma=0.2):
    t = epoch_ / T
    return base - (base - floor) * (t ** gamma)


def auto_range(epoch_, T, base, floor=0.02, decay_type='linear', gamma=0.2):
    if decay_type == 'linear':
        return linear_range(epoch_, T, base, floor)
    elif decay_type == 'polynomial':
        return polynomial_range(epoch_, T, base, floor, gamma)
    else:
        raise ValueError(f'Invalid decay type: {decay_type}')

def try_float(x):
    try:
        return float(x)
    except Exception as e:
        return 0
    
def get_info(rows, columns):
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
    return info

def get_info_gaussian(data, columns):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    info = {}
    for i, c in enumerate(columns):
        info[c] = {
            'min': float(data[:, i].min()),
            'max': float(data[:, i].max())
        }
    return info


def build_columns_info(X, numerical_col_indices, categorical_col_indices):
    num_names = [f"num_{i}" for i in numerical_col_indices]
    cat_names = [f"cat_{i}" for i in categorical_col_indices]
    columns   = {"numerical": num_names, "categorical": cat_names}

    info = {}
    for j, col_idx in enumerate(numerical_col_indices):
        c = num_names[j]
        info[c] = {
            "min": float(X[:, col_idx].min()),
            "max": float(X[:, col_idx].max()),
        }
    for j, col_idx in enumerate(categorical_col_indices):
        c = cat_names[j]
        unique_vals = np.unique(X[:, col_idx])
        info[c] = {v: i for i, v in enumerate(unique_vals)}

    return columns, info


def features_to_samples(X, numerical_col_indices, categorical_col_indices):
    samples = []
    for i in range(len(X)):
        row = [float(X[i, col]) for col in numerical_col_indices]
        row += [X[i, col] for col in categorical_col_indices]
        samples.append(row)
    return samples


def samples_to_features(samples, numerical_col_indices, categorical_col_indices, original_n_features):
    n = len(samples)
    out = np.zeros((n, original_n_features), dtype=float)
    n_num = len(numerical_col_indices)
    for i, s in enumerate(samples):
        for j, col_idx in enumerate(numerical_col_indices):
            out[i, col_idx] = s[j]
        for j, col_idx in enumerate(categorical_col_indices):
            out[i, col_idx] = s[n_num + j]
    return out

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


def variation_sample(sample, columns, info, epoch_, total_epochs,
                     variance_multiplier, decay_type='linear', gamma=0.2):
    s = copy.deepcopy(sample)
    for i, c in enumerate(columns["numerical"] + columns["categorical"]):
        if c in columns["numerical"]:
            multiplier_range = auto_range(epoch_, total_epochs, variance_multiplier,
                                          decay_type=decay_type, gamma=gamma)
            s[i] += random.uniform(-multiplier_range, multiplier_range) * (info[c]['max'] - info[c]['min'])
            s[i] = max(info[c]['min'], min(info[c]['max'], s[i]))
        else:
            probability = auto_range(epoch_, total_epochs, variance_multiplier,
                                     decay_type=decay_type, gamma=gamma)
            if np.random.choice([0, 1], p=[1.0 - probability, probability]):
                s[i] = random.choice(list(info[c].keys()))
    return s

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


def get_embeddings(samples, columns, info):
    return np.array([get_vector(s, columns, info) for s in samples])

def vote(public, public_embeddings, private_embeddings, count, noise_multiplier):
    distances = torch.cdist(
        torch.Tensor(private_embeddings), 
        torch.Tensor(public_embeddings)
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

def generate_random(n, columns, info):
    samples = []
    for _ in range(n):
        row = [random.uniform(info[c]['min'], info[c]['max']) for c in columns["numerical"]]
        row += [random.choice(list(info[c].keys())) for c in columns["categorical"]]
        samples.append(row)
    return samples

def run_voting(generator_fn, random_fn, generator_fraction,
               private_embeddings_per_class, n_per_split_per_class,
               vote_rounds, noise_sigma, oversample_factor, columns, info):

    n_classes = len(private_embeddings_per_class)

    # initialise — generate first batch of winners from scratch
    syn_per_class = {}
    for label in range(n_classes):
        syn_per_class[label] = generate_samples(
            n_per_split_per_class[label], label,
            generator_fn, random_fn, generator_fraction
        )

    for round_i in range(vote_rounds):
        print(f"    [Round {round_i + 1}/{vote_rounds}]")
        for label in range(n_classes):
            n_per_split = n_per_split_per_class[label]

            # generate fresh candidates on top of current winners
            n_fresh = int(oversample_factor * n_per_split)
            fresh = generate_samples(n_fresh, label, generator_fn, random_fn, generator_fraction)

            # pool = current winners + fresh candidates
            pool = syn_per_class[label] + fresh

            # embed pool
            pool_embeddings = get_embeddings(pool, columns, info)

            # vote — keep top n_per_split
            winners, _, _ = vote(pool, pool_embeddings,
                                 private_embeddings_per_class[label],
                                 n_per_split, noise_sigma)
            syn_per_class[label] = winners

    return syn_per_class


# Note when passing in generator_fn and random_fn can pass in like:
# random_fn    = lambda n, label: generate_random(n, columns, info)
# generator_fn = lambda n, label: synthesize_data_for_label(gen, device, label, n)
def run_voting_pipeline(generator_fn, random_fn, generator_fraction,
                        X_train, y_train, n_classes,
                        num_synth_factor, k_splits, vote_rounds,
                        oversample_factor, epsilon_vote,
                        numerical_col_indices, categorical_col_indices):
    """
    Outer wrapper for the full split-vote-union pipeline.

    generator_fn:            callable(n, label) -> list of samples from model 
    random_fn:               callable(n, label) -> list of random samples
    generator_fraction:      float in [0, 1]
    X_train:                 np.array (n_train, n_features)
    y_train:                 np.array (n_train,) integer class labels
    n_classes:               number of classes
    num_synth_factor:        size of final dataset relative to n_train
    k_splits:                number of independent splits to union
    vote_rounds:             refinement rounds per split
    oversample_factor:       fresh candidates per round = oversample_factor * n_per_split
    epsilon_vote:            total DP epsilon budget for all voting steps
    numerical_col_indices:   list of int
    categorical_col_indices: list of int

    Returns:
        final_features: np.array (n_synth, n_features)
        final_labels:   np.array (n_synth,) integer class labels
    """
    n_train = len(X_train)
    n_synth = int(n_train * num_synth_factor)

    # compute per-class counts proportionally to training distribution
    class_counts    = np.array([np.sum(y_train == label) for label in range(n_classes)])
    class_fractions = class_counts / n_train
    n_per_class     = np.floor(class_fractions * n_synth).astype(int)

    # distribute remainder to largest classes
    remainder       = n_synth - n_per_class.sum()
    largest_classes = np.argsort(class_fractions)[::-1]
    for i in range(remainder):
        n_per_class[largest_classes[i]] += 1

    n_per_split_per_class = {label: math.ceil(n_per_class[label] / k_splits)
                            for label in range(n_classes)}

    # warn if voting quality may be poor for any class
    for label in range(n_classes):
        n_candidates = int(n_per_split_per_class[label] * (1 + oversample_factor))
        n_private    = int(class_counts[label])
        if n_candidates > n_private:
            print(f"WARNING: class {label} — candidates per split ({n_candidates}) "
                  f"exceeds private points ({n_private}). "
                  f"Consider reducing oversample_factor or increasing k_splits.")

    print(f"  n_synth:           {n_synth}")
    print(f"  k_splits:          {k_splits}")
    print(f"  n_per_class:       {n_per_class.tolist()}")
    print(f"  n_per_split:       {[n_per_split_per_class[l] for l in range(n_classes)]}")
    print(f"  oversample_factor: {oversample_factor}")

    # build columns and info from training data
    columns, info = build_columns_info(X_train, numerical_col_indices, categorical_col_indices)

    # calibrate noise — num_steps = vote_rounds * k_splits
    noise_sigma = find_required_noise_multiplier(
        epsilon_vote,
        num_steps=vote_rounds * k_splits,
        num_N=n_train
    )
    print(f"  noise_sigma:       {noise_sigma:.4f}")

    # precompute private embeddings once — no privacy cost
    private_embeddings_per_class = {}
    for label in range(n_classes):
        cls_X       = X_train[y_train == label]
        cls_samples = features_to_samples(cls_X, numerical_col_indices, categorical_col_indices)
        private_embeddings_per_class[label] = get_embeddings(cls_samples, columns, info)

    # run k_splits independent pipelines and union results
    all_features, all_labels = [], []

    for split_idx in range(k_splits):
        print(f"  [Split {split_idx + 1}/{k_splits}]")

        syn_per_class = run_voting(
            generator_fn=generator_fn,
            random_fn=random_fn,
            generator_fraction=generator_fraction,
            private_embeddings_per_class=private_embeddings_per_class,
            n_per_split_per_class=n_per_split_per_class,
            vote_rounds=vote_rounds,
            noise_sigma=noise_sigma,
            oversample_factor=oversample_factor,
            columns=columns,
            info=info,
        )

        # collect winners from this split, adding labels back
        for label in range(n_classes):
            features = samples_to_features(
                syn_per_class[label],
                numerical_col_indices,
                categorical_col_indices,
                X_train.shape[1]
            )
            all_features.append(features)
            all_labels.append(np.full(len(features), label, dtype=int))

    final_features = np.concatenate(all_features, axis=0)
    final_labels   = np.concatenate(all_labels,   axis=0)
    
    all_features_out, all_labels_out = [], []
    for label in range(n_classes):
        label_mask = (final_labels == label)
        label_features = final_features[label_mask]
        target = n_per_class[label]
        if len(label_features) > target:
            keep = np.random.choice(len(label_features), size=target, replace=False)
            label_features = label_features[keep]
        all_features_out.append(label_features)
        all_labels_out.append(np.full(len(label_features), label, dtype=int))

    final_features = np.concatenate(all_features_out, axis=0)
    final_labels   = np.concatenate(all_labels_out,   axis=0)
    
    # ===== Sanity check: size correctness =====
    expected_n = n_synth
    actual_n   = final_features.shape[0]
    label_n    = final_labels.shape[0]

    if actual_n != expected_n:
        raise ValueError(
            f"[SIZE ERROR] Expected {expected_n} synthetic samples, got {actual_n}. "
            f"Check k_splits, per-class allocation, or voting truncation."
        )

    if label_n != expected_n:
        raise ValueError(
            f"[LABEL ERROR] Expected {expected_n} labels, got {label_n}."
        )

    if actual_n != label_n:
        raise ValueError(
            f"[ALIGNMENT ERROR] Features/labels mismatch: "
            f"{actual_n} vs {label_n}"
        )

    print(f"[OK] Final synthetic dataset size: {actual_n}")
        
    return final_features, final_labels

def generate_samples(n, label, generator_fn, random_fn, generator_fraction):
    n_from_gen  = int(n * generator_fraction)
    n_from_rand = n - n_from_gen
    
    samples = []
    if n_from_gen > 0:
        samples.extend(generator_fn(n_from_gen, label))
    if n_from_rand > 0:
        samples.extend(random_fn(n_from_rand, label))
    return samples