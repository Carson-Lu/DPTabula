seed=42
epsilon=1.0
degree=2
classifier="xgboost"
device="cuda:0"
bin_type="privtree"
eval_only=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) method="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --epsilon) epsilon="$2"; shift ;;
        --degree) degree="$2"; shift ;;
        --classifier) classifier="$2"; shift ;;
        --device) device="$2"; shift ;;
        --bin_type) bin_type="$2"; shift ;;
        --eval_only) eval_only="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

data_dir="../data/processed/$dataset/$seed"
data_train="$data_dir/data_train.csv"
data_val="$data_dir/data_val.csv"
data_test="$data_dir/data_test.csv"
metadata_path="../data/$dataset/metadata.json"
output_dir="/home/carson/scratch/tabpe_logs/$dataset/$seed/$method/eps_${epsilon}_degree_${degree}_${bin_type}"

bash scripts/data-split.sh --dataset "$dataset" --seed "$seed"

echo -e "\nRunning $method =============================================================="
if [[ "$eval_only" == false ]]; then
# Start timing
start_time=$(date +%s)
start_time_readable=$(date)

echo "Starting $method execution at: $start_time_readable"

python -u ../src/model/baselines/main.py \
    --method $method \
    --device $device \
    --epsilon $epsilon \
    --degree $degree \
    --priv_data_dir data/processed/$dataset/$seed \
    --metadata_path data/$dataset/metadata.json \
    --output_dir $output_dir \
    --num_preprocess $bin_type

# End timing
end_time=$(date +%s)
end_time_readable=$(date)

# Calculate running time
running_time=$((end_time - start_time))
running_time_minutes=$((running_time / 60))
running_time_seconds=$((running_time % 60))

echo "$method execution completed at: $end_time_readable"
echo "Total running time: ${running_time_minutes}m ${running_time_seconds}s (${running_time} seconds)"

# Save timing information to output directory
cat > "$output_dir/timing.txt" << EOF
$method Execution Timing
==================

Start time: $start_time_readable
End time: $end_time_readable
Total running time: ${running_time_minutes}m ${running_time_seconds}s (${running_time} seconds)

Parameters:
- Dataset: $dataset
- Epsilon: $epsilon
- Degree: $degree
- Bin type: $bin_type
EOF

echo "Timing information saved to: $output_dir/timing.txt"
fi

echo -e "\nRunning evaluation =============================================================="

python -u ../src/evaluation/eval.py \
    --epochs 0 \
    --metadata_path $metadata_path \
    --priv_train_csv $data_train \
    --priv_val_csv $data_val \
    --priv_test_csv $data_test \
    --synthetic_data_dir $output_dir \
    --classifier $classifier

python -u ../src/evaluation/eval_embedding.py \
    --dataset $dataset \
    --seed $seed \
    --synthetic_dir $output_dir