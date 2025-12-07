import os
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd
from collections import Counter
import torch
from itertools import combinations
import logging
import sys
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from prdc import compute_prdc
import numpy as np
from tqdm import tqdm
from src.utils import get_embeddings, get_info

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def run_classifier(df_train, df_test, columns, classifier='xgboost'):
    # with open(args.path_columns, 'r') as f:
    #     columns = json.load(f)

    # df_train = pd.read_csv(path_train)
    y_train = df_train[columns['label']].values.ravel()
    X_train = df_train.drop(columns['label'], axis=1)

    # df_test = pd.read_csv(path_test)
    y_test = df_test[columns['label']].values.ravel()
    X_test = df_test.drop(columns['label'], axis=1)

    transformer = ColumnTransformer(transformers=[('ordinal', OrdinalEncoder(), columns['categorical'])], remainder='passthrough')
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.fit_transform(X_test)

    labelencoder = LabelEncoder()
    y_train = labelencoder.fit_transform(y_train)
    y_test = labelencoder.fit_transform(y_test)

    # defaults to multi:softmax if number of classes > 2
    # model = XGBClassifier(objective='binary:logistic', random_state=args.seed, n_jobs=1)
    num_classes = len(np.unique(y_train))
    if classifier == 'xgboost':
        if num_classes == 2:
            model = XGBClassifier(
                objective='binary:logistic',
                tree_method='hist', 
                device='cuda',
                random_state=42,
                n_jobs=8
            )
        else:
            model = XGBClassifier(
                objective='multi:softmax',
                tree_method='hist',
                num_class=num_classes,
                device='cuda', # Use GPU if available
                random_state=42,
                n_jobs=8
        )
    elif classifier == 'linear':
        model = LogisticRegression(random_state=42, n_jobs=1)
    elif classifier == 'tabpfn':
        model = TabPFNClassifier()
        if len(X_train) > 10000:
            # Set random seed for reproducibility
            np.random.seed(42)
            # Randomly sample 10000 indices
            indices = np.random.choice(len(X_train), 10000, replace=False)
            X_train, y_train = X_train[indices], y_train[indices]
    elif classifier == 'tabicl':
        model = TabICLClassifier()
        if len(X_train) > 60000:
            np.random.seed(42)
            indices = np.random.choice(len(X_train), 60000, replace=False)
            X_train, y_train = X_train[indices], y_train[indices]
    else:
        raise ValueError(f"Model {model} not supported")
        
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # check if y_pred is binary
    if len(np.unique(y_test)) == len(np.unique(y_pred)) == 2:
        y_pred_proba = model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc_score = -1.0

    return accuracy_score(y_test, y_pred), auc_score, f1_score(y_test, y_pred, average='macro')


def average_k_tvd(df_syn, df_test, columns, K=2, num_bins=20):
    df1 = df_test.copy()
    df2 = df_syn.copy()
    all_cols = columns["numerical"] + columns["categorical"] + [columns["label"]]

    for col in columns["numerical"]:
        edges = np.linspace(df1[col].min(), df1[col].max(), num_bins)
        df1[col] = pd.cut(df1[col], bins=edges, include_lowest=True).astype('category')
        df2[col] = pd.cut(df2[col], bins=edges, include_lowest=True).astype('category')

    for col in columns["categorical"] + [columns["label"]]:
        all_categories = df1[col].dropna().unique()
        cat_type = pd.api.types.CategoricalDtype(categories=all_categories, ordered=True)
        df1[col] = df1[col].astype(cat_type)
        df2[col] = df2[col].astype(cat_type)

    combos = list(combinations(all_cols, K))
    # combos = [c for c in combos if columns["label"] in c] # only include combos that include label

    if not combos:
        return 0.0

    tvds = []
    for group in combos:
        group_list = list(group)
        p = df1.value_counts(subset=group_list, normalize=True, sort=False)
        q = df2.value_counts(subset=group_list, normalize=True, sort=False)
        union_index = p.index.union(q.index)        
        p_aligned = p.reindex(union_index, fill_value=0.0)
        q_aligned = q.reindex(union_index, fill_value=0.0)
        tvd = 0.5 * (p_aligned - q_aligned).abs().sum()
        tvds.append(tvd)

    return sum(tvds) / len(tvds)


def get_prdc(df_syn, df_priv_train, info, columns, nearest_k=5):
    syn_embeddings = get_embeddings(df_syn, info, columns)
    priv_train_embeddings = get_embeddings(df_priv_train, info, columns)
    prdc = compute_prdc(real_features=priv_train_embeddings, fake_features=syn_embeddings, nearest_k=nearest_k)
    return prdc

def main(args):
    synthetic_data_dir = args.synthetic_data_dir
    classifier = args.classifier
    nearest_k = 5
    path_results = f"{synthetic_data_dir}/results_{classifier}.json"
    if os.path.exists(path_results):
        with open(path_results, 'r') as f:
            results = json.load(f)
        print(f"Results already exist: {results}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        filename=f"{synthetic_data_dir}/eval_{classifier}.log",
        filemode='w'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'))
    logging.getLogger().addHandler(console)

    columns = json.load(open(args.metadata_path, 'r'))
    best_accuracy = -1.0
    df_priv = pd.read_csv(args.priv_train_csv)
    df_val = pd.read_csv(args.priv_val_csv)
    df_test = pd.read_csv(args.priv_test_csv)
    df_priv_all = pd.concat([df_priv, df_val, df_test])
    info = get_info(df_priv_all, columns)

    csv_name = "synthetic_df.csv"
    accuracies = []
    for epoch in tqdm(range(args.epochs + 1)):
        df_train = pd.read_csv(f"{synthetic_data_dir}/{epoch}/{csv_name}")
        path_PE = f"{synthetic_data_dir}/{epoch}/{csv_name}"
        accuracy, roc_auc, f1 = run_classifier(df_train, df_val, columns, classifier)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            path_PE_best = path_PE
        avg_1_tvd = average_k_tvd(df_train, df_priv, columns, K=1, num_bins=20)
        avg_2_tvd = average_k_tvd(df_train, df_priv, columns, K=2, num_bins=20)
        avg_3_tvd = average_k_tvd(df_train, df_priv, columns, K=3, num_bins=20)
        logging.info(f"Epoch {epoch} -- val accuracy: {accuracy:.10f}, roc_auc: {roc_auc:.10f}, macro f1: {f1:.10f}, average 1-TVD: {avg_1_tvd:.3f}, average 2-TVD: {avg_2_tvd:.3f}, average 3-TVD: {avg_3_tvd:.3f}")
        if args.epoch_test_acc:
            accuracy_test, roc_auc_test, f1_test = run_classifier(df_train, df_test, columns, classifier)
            logging.info(f"\tTest accuracy: {accuracy_test:.10f}, roc_auc: {roc_auc_test:.10f}, macro f1: {f1_test:.10f}")

    plt.plot(range(args.epochs + 1), accuracies)
    plt.savefig(f"{synthetic_data_dir}/accuracies_validation.png")
    plt.close()

    # last epoch test accuracy
    df_pe_last = pd.read_csv(f"{synthetic_data_dir}/{args.epochs}/{csv_name}")
    accuracy_test_last, roc_auc_test_last, f1_test_last = run_classifier(df_pe_last, df_test, columns, classifier)
    logging.info(f"Last epoch test accuracy: {accuracy_test_last:.10f}, roc_auc: {roc_auc_test_last:.10f}, macro f1: {f1_test_last:.10f}")
    if args.epoch_prdc:
        for label in df_priv[columns['label']].unique():
            df_priv_label = df_priv[df_priv[columns['label']] == label]
            df_syn_label = df_pe_last[df_pe_last[columns['label']] == label]
            if (len(df_syn_label) <= nearest_k) or (len(df_priv_label) <= nearest_k):
                continue
            prdc = get_prdc(df_syn_label, df_priv_label, info, columns, nearest_k=nearest_k)
            logging.info(f"\tLabel {label} PRDC: {prdc}")

    df_pe_best = pd.read_csv(path_PE_best)
    accuracy_test, roc_auc_test, f1_test = run_classifier(df_pe_best, df_test, columns, classifier)
    avg_1_tvd = average_k_tvd(df_pe_best, df_priv, columns, K=1, num_bins=20)
    avg_2_tvd = average_k_tvd(df_pe_best, df_priv, columns, K=2, num_bins=20)
    avg_3_tvd = average_k_tvd(df_pe_best, df_priv, columns, K=3, num_bins=20)
    logging.info(f"Best Val: Test accuracy using best epoch according to val: {accuracy_test:.10f}, roc_auc: {roc_auc_test:.10f}, macro f1: {f1_test:.10f}, average 1-TVD: {avg_1_tvd:.3f}, average 2-TVD: {avg_2_tvd:.3f}, average 3-TVD: {avg_3_tvd:.3f}")

    accuracy_train, roc_auc_train, f1_train =  run_classifier(df_priv, df_test, columns, classifier)
    avg_1_tvd = average_k_tvd(df_test, df_priv, columns, K=1, num_bins=20)
    avg_2_tvd = average_k_tvd(df_test, df_priv, columns, K=2, num_bins=20)
    avg_3_tvd = average_k_tvd(df_test, df_priv, columns, K=3, num_bins=20)
    logging.info(f"Upper Bound (using priv train data): Test accuracy: {accuracy_train:.10f}, roc_auc: {roc_auc_train:.10f}, macro f1: {f1_train:.10f}, average 1-TVD: {avg_1_tvd:.3f}, average 2-TVD: {avg_2_tvd:.3f}, average 3-TVD: {avg_3_tvd:.3f}")

    if args.epoch_prdc:
        for label in df_priv[columns['label']].unique():
            df_priv_label = df_priv[df_priv[columns['label']] == label]
            df_test_label = df_test[df_test[columns['label']] == label]
            if (len(df_test_label) <= nearest_k) or (len(df_priv_label) <= nearest_k):
                continue
            prdc = get_prdc(df_test_label, df_priv_label, info, columns, nearest_k=nearest_k)
            logging.info(f"\tLabel {label} PRDC: {prdc}")

    t = Counter(df_test[columns['label']])
    majority_class = max(t, key=t.get)
    lower_bound_pred = [majority_class] * len(df_test)
    lower_bound_accuracy = accuracy_score(lower_bound_pred, df_test[columns['label']])
    lower_bound_f1 = f1_score(lower_bound_pred, df_test[columns['label']], average='macro')
    logging.info(f"Lower Bound: Trivial accuracy: {lower_bound_accuracy:.10f}, macro f1: {lower_bound_f1:.10f}")

    j = {}
    j['lower_bound'] = {
        'accuracy': lower_bound_accuracy,
        'roc_auc': 0.5,
        'f1': lower_bound_f1
    }
    j['upper_bound'] = {
        'accuracy': accuracy_train,
        'roc_auc': roc_auc_train,
        'f1': f1_train
    }
    j['best_val'] = {
        'accuracy': accuracy_test,
        'roc_auc': roc_auc_test,
        'f1': f1_test
    }
    j['last_epoch'] = {
        'accuracy': accuracy_test_last,
        'roc_auc': roc_auc_test_last,
        'f1': f1_test_last
    }

    with open(path_results, 'w') as f:
        json.dump(j, f)

    print(f"Results saved to {path_results}")


def parse_args():
    parser = argparse.ArgumentParser(description='PE for tabular data')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--metadata_path', default=None, type=str, help='Path to metadata file')
    parser.add_argument('--priv_train_csv', default=None, type=str, help='Path to private training data file')
    parser.add_argument('--priv_val_csv', default=None, type=str, help='Path to private validation data file')
    parser.add_argument('--priv_test_csv', default=None, type=str, help='Path to private test data file')
    parser.add_argument('--synthetic_data_dir', default=None, type=str, help='Path to synthetic data directory')
    parser.add_argument('--classifier', default='xgboost', type=str, help='Classifier to use')
    parser.add_argument('--epoch-prdc', action='store_true', help='Enable PRDC for each epoch')
    parser.add_argument('--epoch-test-acc', action='store_true', help='Enable test accuracy for each epoch')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
