import os 
import numpy as np
import pandas as pd
import argparse
import math
import json
import time
from copy import deepcopy, copy
from typing import Union
from util.util import * 
from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho
from evaluator.eval_seeds import eval_seeds
from evaluator.eval_sample import eval_sampler
from pathlib import Path
import tempfile
import shutil
import os

description = ""
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
parser.add_argument("--dataset", default='placeholder', help="dataset to use")
parser.add_argument("--method", help="synthesis method")
parser.add_argument("--device", help="device to use")
parser.add_argument("--epsilon", type=float, help="privacy parameter")
parser.add_argument("--priv_data_dir", help="private data directory")
parser.add_argument("--metadata_path", help="metadata path")
parser.add_argument("--output_dir", help="synthetic data directory")
# parser.add_argument("--delta", type=float, default=1e-5, help="privacy parameter")
parser.add_argument("--degree", type=int, default=2, help="degree of the model")
parser.add_argument("--num_preprocess", type=str, default='uniform_kbins')
parser.add_argument("--rare_threshold", type=float, default=0.002) # if 0 then 3sigma
parser.add_argument("--sample_device", help="device to synthesis, only used in some deep learning models", default=None)
parser.add_argument("--test", action="store_true", help="test mode (placeholder only, don't use)")
args = parser.parse_args()

if args.sample_device is None:
    args.sample_device = args.device

if args.method in ['rap', 'rap_syn']:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

def df_to_npy(priv_data_dir, metadata_path, npy_output_dir):
    df_train = pd.read_csv(os.path.join(priv_data_dir, 'data_train.csv'))
    df_val = pd.read_csv(os.path.join(priv_data_dir, 'data_val.csv'))
    df_test = pd.read_csv(os.path.join(priv_data_dir, 'data_test.csv'))
    # df_all = pd.concat([df_train, df_val, df_test])
    metadata = json.load(open(metadata_path))

    npy_output_dir = Path(npy_output_dir)
    npy_output_dir.mkdir(exist_ok=True)

    if len(metadata['numerical']) > 0:
        X_num_train = df_train[metadata['numerical']].values
        X_num_val = df_val[metadata['numerical']].values
        X_num_test = df_test[metadata['numerical']].values
        np.save(npy_output_dir / 'X_num_train.npy', X_num_train)
        np.save(npy_output_dir / 'X_num_val.npy', X_num_val)
        np.save(npy_output_dir / 'X_num_test.npy', X_num_test)


    if len(metadata['categorical']) > 0:
        X_cat_train = df_train[metadata['categorical']].values
        X_cat_val = df_val[metadata['categorical']].values
        X_cat_test = df_test[metadata['categorical']].values
        np.save(npy_output_dir / 'X_cat_train.npy', X_cat_train)
        np.save(npy_output_dir / 'X_cat_val.npy', X_cat_val)
        np.save(npy_output_dir / 'X_cat_test.npy', X_cat_test)


    # convert target to int
    target_col = metadata['label']
    target_map = {v: i for i, v in enumerate(df_train[target_col].unique())}
    df_train[target_col] = df_train[target_col].map(target_map)
    df_val[target_col] = df_val[target_col].map(target_map)
    df_test[target_col] = df_test[target_col].map(target_map)

    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values

    # create npy output dir with Pathlib
    np.save(npy_output_dir / 'y_train.npy', y_train)
    np.save(npy_output_dir / 'y_val.npy', y_val)
    np.save(npy_output_dir / 'y_test.npy', y_test)
    
    # create domain.json
    domain = {}
    for i, col in enumerate(metadata['numerical']):
        domain[f"num_attr_{i+1}"] = df_train[col].nunique()
    for i, col in enumerate(metadata['categorical']):
        domain[f"cat_attr_{i+1}"] = df_train[col].nunique()
    domain['y_attr'] = df_train[target_col].nunique()
    with open(npy_output_dir / 'domain.json', 'w') as f:
        json.dump(domain, f)

    return target_map, len(df_train)

def npy_to_df(synthetic_output_dir, target_map, metadata):
    target_col = metadata['label']
    X_num = None
    X_cat = None
    y = None
    if len(metadata['numerical']) > 0:
        X_num = np.load(synthetic_output_dir / 'X_num_train.npy')
    if len(metadata['categorical']) > 0:
        X_cat = np.load(synthetic_output_dir / 'X_cat_train.npy')
    y = np.load(synthetic_output_dir / 'y_train.npy')
    if X_num is not None and X_cat is not None:
        df = pd.DataFrame(np.concatenate([X_num, X_cat], axis=1), columns=metadata['numerical'] + metadata['categorical'])
    elif X_num is not None:
        df = pd.DataFrame(X_num, columns=metadata['numerical'])
    elif X_cat is not None:
        df = pd.DataFrame(X_cat, columns=metadata['categorical'])
    else:
        raise ValueError("No numerical or categorical columns found")
    df[target_col] = y.astype(int)
    # target map is a dict of {target_cat: target_index}
    invert_target_map = {v: k for k, v in target_map.items()}
    df[target_col] = df[target_col].map(invert_target_map)
    
    return df

def main(args):
    # check if output synthetic_df.csv exists
    if os.path.exists(f"{args.output_dir}/0/synthetic_df.csv"):
        print(f"Synthetic data already exists in {args.output_dir}/0/synthetic_df.csv")
        return
    
    # create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix=f"tabbench"))
    temp_dir.mkdir(exist_ok=True)
    data_path = temp_dir / 'npy'
    data_path.mkdir(exist_ok=True)
    parent_dir = temp_dir / 'exp'
    parent_dir.mkdir(exist_ok=True)

    
    # convert df to npy
    target_map, num_N = df_to_npy(args.priv_data_dir, args.metadata_path, data_path)
    args.delta = 1/(num_N*math.log(num_N))
    print(f'privacy setting: ({args.epsilon}, {args.delta})')


    # parent_dir, data_path = make_exp_dir(args)
    time_record = {}

    # data preprocess
    total_rho = cdp_rho(args.epsilon, args.delta)
    data_preprocesser = data_preporcesser_common(args)
    df, domain, preprocesser_divide  = data_preprocesser.load_data(data_path, total_rho)

    if args.method == 'ddpm':
        param_dict = {'rho_used': preprocesser_divide*total_rho} 
    else:
        param_dict = {}
    

    # fitting model
    start_time = time.time()
    generator_dict = algo_method(args)(
        args, df=df, domain=domain, 
        rho=(1-preprocesser_divide)*total_rho, 
        parent_dir=parent_dir, 
        preprocesser = data_preprocesser,
        **param_dict
    )
    end_time = time.time()
    time_record['model fitting time'] = end_time-start_time

    # evaluation
    # eval_config = prepare_eval_config(args, parent_dir)

    eval_config = {
        'parent_dir': str(parent_dir),  # Convert Path to string for JSON serialization
        'sample': {'seed': 0, 'sample_num': len(df)}
    }

    # Create a permanent directory for synthetic data
    synthetic_dir = parent_dir / "synthetic_data"
    synthetic_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving synthetic data to: {synthetic_dir}")
    
    # Create a copy of eval_config with synthetic_dir as the target
    synthesis_config = copy(eval_config)
    synthesis_config['parent_dir'] = str(synthetic_dir)
    
    # Run synthesis directly to the synthetic_data directory
    eval_sampler(args.method, synthesis_config, args.sample_device, data_preprocesser, **generator_dict)
    
    print(f"Synthesis completed. Data saved to: {synthetic_dir}")
    
    # covert synthetic npy to df
    metadata = json.load(open(args.metadata_path))
    df = npy_to_df(parent_dir / 'synthetic_data', target_map, metadata)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # save df to csv
    epoch_dir = output_dir / '0'
    epoch_dir.mkdir(exist_ok=True, parents=True)

    df.to_csv(epoch_dir / 'synthetic_df.csv', index=False)

    # clean up temporary directory
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main(args)

