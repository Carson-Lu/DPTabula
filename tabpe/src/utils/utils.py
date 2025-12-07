import numpy as np


def try_float(x):
    try:
        return float(x)
    except Exception as e:
        return 0

def get_vector(s, columns, info):
    vector = []

    for i, c in enumerate(columns["numerical"] + columns["categorical"]):
        if c in columns["numerical"]:
            vector.append((s[c] - info[c]['min']) / (info[c]['max'] - info[c]['min']))
        else:
            # Create a mapping from category values to their indices
            category_values = list(info[c].keys())
            vector_add = [0.0] * len(info[c])
            if str(s[c]) in info[c]:
                # Find the index of this category value
                category_index = category_values.index(str(s[c]))
                vector_add[category_index] = 1.0 / 3.0
            else:
                print(f"Unseen categorical value {str(s[c])} in column {c}")
            vector += vector_add
    return vector

def get_info(df, columns):
    info = {}
    for c in columns["numerical"]:
        info[c] = {'min': df[c].min(), 'max': df[c].max()}
    for c in columns["categorical"]:
        info[c] = df[c].value_counts().to_dict()
    return info

def get_embeddings(df, info, columns):
    embeddings = np.array([get_vector(row, columns, info) for _, row in df.iterrows()])

    return embeddings
