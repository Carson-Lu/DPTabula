import copy
import random
import torch as pt
import numpy as np
from collections import defaultdict


# ------------------------------------------------------------------ #
#  Variance decay schedules                                           #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Embeddings                                                         #
# ------------------------------------------------------------------ #

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
    """
    data: pt.Tensor or np.array of shape (n_samples, n_features)
    columns: list of column names e.g. ['x', 'y']
    """
    if isinstance(data, pt.Tensor):
        data = data.cpu().numpy()
    
    info = {}
    for i, c in enumerate(columns):
        info[c] = {
            'min': float(data[:, i].min()),
            'max': float(data[:, i].max())
        }
    return info

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


def get_embeddings(samples, columns, info):
    return np.array([get_vector(s, columns, info) for s in samples], dtype=np.float32)


# ------------------------------------------------------------------ #
#  Variation sampling                                                 #
# ------------------------------------------------------------------ #

def variation_sample(sample, columns, info, epoch_, total_epochs,
                     variance_multiplier, decay_type='linear', gamma=0.2):
    """
    Generate a perturbed variation of a single sample.

    Args:
        sample:             list of feature values
        columns:            dict with keys 'numerical', 'categorical'
        info:               per-column min/max/categories dict
        epoch_:             current epoch
        total_epochs:       total number of epochs
        variance_multiplier: base variance scale
        decay_type:         'linear' or 'polynomial'
        gamma:              decay exponent for polynomial schedule
    """
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


# ------------------------------------------------------------------ #
#  Random initialization                                              #
# ------------------------------------------------------------------ #

def random_init(counts, columns, info):
    """
    Generate random samples uniformly over the feature space.
    Works for any tabular data including 2D (x, y treated as numerical).

    Args:
        counts:  dict {label: n_samples}
        columns: dict with keys 'numerical', 'categorical'
        info:    per-column min/max/categories dict

    Returns:
        public:  defaultdict {label: [sample, ...]}
    """
    def random_sample():
        numerical   = [random.uniform(info[c]['min'], info[c]['max'])
                       for c in columns["numerical"]]
        categorical = [random.choice(list(info[c].keys()))
                       for c in columns["categorical"]]
        return numerical + categorical

    public = defaultdict(list)
    for label, n in counts.items():
        for _ in range(n):
            public[label].append(random_sample())
    return public


# ------------------------------------------------------------------ #
#  Core vote function                                                 #
# ------------------------------------------------------------------ #

def vote(public, public_embeddings, private_embeddings, count,
         noise_multiplier, chunk_size=1024):
    """
    Each private point votes for its nearest public candidate.
    Returns top `count` candidates by noisy vote count.

    Args:
        public:             list of raw samples (any format, returned as-is)
        public_embeddings:  np.array (n_public, d)
        private_embeddings: np.array (n_private, d)
        count:              number of top candidates to return
        noise_multiplier:   std of Gaussian noise added to histogram
        chunk_size:         number of private points processed per GPU batch

    Returns:
        public_best:      list of top `count` samples
        histogram:        raw vote counts np.array (n_public,)
        noisy_histogram:  noisy vote counts np.array (n_public,)
    """
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    public_t = pt.from_numpy(public_embeddings).to(device=device, dtype=pt.float32)
    num_public = len(public_embeddings)
    histogram = np.zeros(num_public, dtype=np.int64)

    for i in range(0, len(private_embeddings), chunk_size):
        private_chunk = pt.from_numpy(
            private_embeddings[i:i + chunk_size]
        ).to(device=device, dtype=pt.float32)
        distances = pt.cdist(private_chunk, public_t)
        votes = distances.argmin(dim=1).cpu().numpy()
        np.add.at(histogram, votes, 1)
        del distances, private_chunk

    noisy_histogram = histogram + np.random.normal(0, noise_multiplier, size=num_public)
    top_indices = np.argsort(noisy_histogram)[::-1][:count]
    public_best = [public[i] for i in top_indices]

    del public_t
    pt.cuda.empty_cache()

    return public_best, histogram, noisy_histogram


# ------------------------------------------------------------------ #
#  Vote wrapper — works for any synthetic + real data                 #
# ------------------------------------------------------------------ #

def vote_on_synthetic(public, private, count, noise_multiplier, columns, info):
    """
    Full voting pipeline: embed → vote → return best samples.
    Works for any tabular format including 2D Gaussian blobs.

    Args:
        public:           list of synthetic samples to vote on
        private:          list of real (private) samples that cast votes
        count:            number of samples to keep
        noise_multiplier: DP noise std
        columns:          dict with keys 'numerical', 'categorical'
        info:             per-column min/max/categories dict

    Returns:
        public_best:      list of top `count` synthetic samples
        histogram:        raw vote counts
        noisy_histogram:  noisy vote counts
    """
    public_embeddings  = get_embeddings(public,  columns, info)
    private_embeddings = get_embeddings(private, columns, info)

    return vote(public, public_embeddings, private_embeddings,
                count, noise_multiplier)