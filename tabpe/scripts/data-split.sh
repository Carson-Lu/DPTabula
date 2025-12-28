#!/bin/bash

#Default values
dataset="adult"
seed="42"

# Usage: ./run_data_split.sh --dataset adult --seed 42

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

[[ -z "$dataset" || -z "$seed" ]] && { echo "Usage: --dataset <name> --seed <seed>"; exit 1; }

data_folder="/home/carson/scratch/data_tabpe/$dataset"
data_all="$data_folder/${dataset}.csv"

python -u ../src/data_processing/main.py \
    --data_all "$data_all" \
    --output_dir "$data_folder/processed/$dataset" \
    --seed "$seed"