#!/bin/bash
#SBATCH --job-name=pe_tabula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1 
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/pe_tabula_%j.out
#SBATCH --error=/home/carson/scratch/logs/pe_tabula_%j.err

# ----- Default parameters -----
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
BATCH_SIZE=4

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
        --decay_type) decay_type="$2"; shift ;; 
        --gamma) gamma="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_DIR=/home/carson/DPTabula/tabpe
DATA_DIR=/home/carson/scratch/data_tabpe
DATA_SRC="${DATA_DIR}/${dataset}"
MODEL_SRC=/home/carson/scratch/hf_models/tabula-8b
RESULTS_DIR=/home/carson/scratch/Tabpe_results

TMP_PROJECT_DIR=${SLURM_TMPDIR}/tabpe
TMP_DATA_DIR=${TMP_PROJECT_DIR}/data
TMP_MODEL_DIR=${TMP_PROJECT_DIR}/tabula-8b
TMP_OUTPUT_DIR=${TMP_PROJECT_DIR}/outputs

module purge
module load python/3.11
source ~/tabpe/ENV/bin/activate

cd "${SLURM_TMPDIR}" || exit
echo "Current directory: ${SLURM_TMPDIR}"

rsync -a --exclude=".git" "${PROJECT_DIR}/" "${TMP_PROJECT_DIR}/"
cd "${TMP_PROJECT_DIR}" || exit
echo "Project copied to ${PWD}"

mkdir -p "${TMP_DATA_DIR}"
rsync -a "${DATA_SRC}/" "${TMP_DATA_DIR}/"
echo "Data copied to ${TMP_DATA_DIR}"

rsync -a "${MODEL_SRC}/" "${TMP_MODEL_DIR}/"
echo "Model copied to ${TMP_MODEL_DIR}"

mkdir -p "${TMP_OUTPUT_DIR}"
OUTPUT_DIR="${RESULTS_DIR}/${dataset}/${seed}/pe/eps_${epsilon}_ns_${num_samples}_e_${epochs}_se_${sampling_epochs}_v_${num_variations}_vm_${variance_multiplier}_dt_${decay_type}_gamma_${gamma}"
mkdir -p "$OUTPUT_DIR"

echo -e "\nRunning data split ========================================================"
bash "${TMP_PROJECT_DIR}/scripts/data-split.sh" --dataset "$dataset" --seed "$seed"

# ----- Run PE -----
if [[ "$eval_only" == false ]]; then
    echo -e "\nRunning PE ==============================================================="
    start_time=$(date +%s)
    start_time_readable=$(date)
    echo "Starting PE execution at: $start_time_readable"

    python -u "${TMP_PROJECT_DIR}/src/model/pe/rw_main_tabula.py" \
        --epochs "$epochs" \
        --sampling_epochs "$sampling_epochs" \
        --priv_train_csv "${TMP_PROJECT_DIR}/scripts/data/processed/${dataset}/${seed}/data_train.csv" \
        --metadata_path "${TMP_DATA_DIR}/${dataset}/metadata.json" \
        --num_samples "$num_samples" \
        --num_variations "$num_variations" \
        --variance_multiplier "$variance_multiplier" \
        --decay_type "$decay_type" \
        --gamma "$gamma" \
        --output_dir "/home/carson/scratch/logs" \
        --epsilon "$epsilon" \
        --model_path "${TMP_MODEL_DIR}" \
        --results_path "${OUTPUT_DIR}/pe_tabula_result_${SLURM_JOB_ID}.txt" \
        --batch_size ${BATCH_SIZE}

    end_time=$(date +%s)
    end_time_readable=$(date)
    running_time=$((end_time - start_time))
    running_time_minutes=$((running_time / 60))
    running_time_seconds=$((running_time % 60))

    echo "PE execution completed at: $end_time_readable"
    echo "Total running time: ${running_time_minutes}m ${running_time_seconds}s (${running_time} seconds)"

    # Save timing info
    cat > "$OUTPUT_DIR/timing.txt" << EOF
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

# ----- Run evaluation -----
python -u "${TMP_PROJECT_DIR}/src/evaluation/eval.py" \
    --epochs "$epochs" \
    --metadata_path "${TMP_DATA_DIR}/${dataset}/metadata.json" \
    --priv_train_csv "${TMP_PROJECT_DIR}/data/processed/${dataset}/${seed}/data_train.csv" \
    --priv_val_csv "${TMP_DATA_DIR}/processed/${dataset}/${seed}/data_val.csv" \
    --priv_test_csv "${TMP_DATA_DIR}/processed/${dataset}/${seed}/data_test.csv" \
    --synthetic_data_dir "$TMP_OUTPUT_DIR" \
    --classifier "$classifier"

python -u "${TMP_PROJECT_DIR}/src/evaluation/eval_embedding.py" \
    --dataset "$dataset" \
    --seed "$seed" \
    --synthetic_dir "$TMP_OUTPUT_DIR"

echo "Done! Results saved to $OUTPUT_DIR"
