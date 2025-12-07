#!/bin/bash

# Default values
dataset="adult"
epochs=15
sampling_epochs=5
num_samples=1000
num_variations=3
variance_multiplier=0.5
epsilon=1.0
seed=42
classifier="tabicl"
eval_only=false
decay_type="polynomial"
gamma=0.2
# Usage: ./run_model.sh [--dataset <name>] [--epochs <n>] [--num_samples <n>] [--num_variations <n>] [--epsilon <val>] [--seed <val>]

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        --sampling_epochs) sampling_epochs="$2"; shift ;;
        --num_samples) num_samples="$2"; shift ;;
        --num_variations) num_variations="$2"; shift ;;
        --variance_multiplier) variance_multiplier="$2"; shift ;;
        --epsilon) epsilon="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --classifier) classifier="$2"; shift ;;
        --eval_only) eval_only="$2"; shift ;;
        --sus) sus="$2"; shift ;;
        --decay_type) decay_type="$2"; shift ;; 
        --gamma) gamma="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

data_dir="data/processed/$dataset/$seed"
data_all="$data_dir/data_train.csv"
data_train="$data_dir/data_train.csv"
data_val="$data_dir/data_val.csv"
data_test="$data_dir/data_test.csv"
metadata_path="data/$dataset/metadata.json"
output_dir="outputs/${dataset}/$seed/pe/eps_${epsilon}_ns_${num_samples}_e_${epochs}_se_${sampling_epochs}_v_${num_variations}_vm_${variance_multiplier}_dt_${decay_type}_gamma_${gamma}"

echo -e "\nRunning data split ========================================================"
bash scripts/data-split.sh --dataset "$dataset" --seed "$seed"

echo -e "\nRunning PE ==============================================================="
# Create output directory if it doesn't exist
mkdir -p "$output_dir"

if [[ "$eval_only" == false ]]; then
# Start timing
start_time=$(date +%s)
start_time_readable=$(date)

echo "Starting PE execution at: $start_time_readable"

PYTHONPATH=. python src/model/pe/rw_main.py \
    --epochs "$epochs" \
    --sampling_epochs "$sampling_epochs" \
    --priv_train_csv "$data_train" \
    --metadata_path "$metadata_path" \
    --num_samples "$num_samples" \
    --num_variations "$num_variations" \
    --variance_multiplier "$variance_multiplier" \
    --decay_type "$decay_type" \
    --gamma "$gamma" \
    --output_dir "$output_dir" \
    --epsilon "$epsilon"


# End timing
end_time=$(date +%s)
end_time_readable=$(date)

# Calculate running time
running_time=$((end_time - start_time))
running_time_minutes=$((running_time / 60))
running_time_seconds=$((running_time % 60))

echo "PE execution completed at: $end_time_readable"
echo "Total running time: ${running_time_minutes}m ${running_time_seconds}s (${running_time} seconds)"

# Save timing information to output directory
cat > "$output_dir/timing.txt" << EOF
PE Execution Timing
==================

Start time: $start_time_readable
End time: $end_time_readable
Total running time: ${running_time_minutes}m ${running_time_seconds}s (${running_time} seconds)

Parameters:
- Dataset: $dataset
- Epochs: $epochs
- Num samples: $num_samples
- Num variations: $num_variations
- Epsilon: $epsilon
- Seed: $seed
EOF

echo "Timing information saved to: $output_dir/timing.txt"
fi

PYTHONPATH=. python src/evaluation/eval.py \
    --epochs $epochs \
    --metadata_path $metadata_path \
    --priv_train_csv $data_train \
    --priv_val_csv $data_val \
    --priv_test_csv $data_test \
    --synthetic_data_dir $output_dir \
    --classifier $classifier

PYTHONPATH=. python src/evaluation/eval_embedding.py \
    --dataset $dataset \
    --seed $seed \
    --synthetic_dir $output_dir