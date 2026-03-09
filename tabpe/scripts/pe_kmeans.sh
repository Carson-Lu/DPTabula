#!/bin/bash
#SBATCH --job-name=pe_kmeans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a100:1 
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/pe_kmeans_%j.out
#SBATCH --error=/home/carson/scratch/logs/pe_kmeans_%j.err

# ----- Default parameters -----
dataset="adult"
epochs=20
num_samples=1000
variance_multiplier=0.5
epsilon=1
seed=42
classifier="linear"
eval_only=false
decay_type="polynomial"
gamma=0.2
BATCH_SIZE=1
generator_method="tabpe"
compare_method="tabula"
num_clusters=5
per_class_kmeans=true # FALSE means that keamns is done on whole dataset

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        --num_samples) num_samples="$2"; shift ;;
        --variance_multiplier) variance_multiplier="$2"; shift ;;
        --epsilon) epsilon="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --classifier) classifier="$2"; shift ;;
        --eval_only) eval_only="$2"; shift ;;
        --decay_type) decay_type="$2"; shift ;; 
        --gamma) gamma="$2"; shift ;;
        --generator_method) generator_method="$2"; shift ;;
        --compare_method) compare_method="$2"; shift ;;
        --num_clusters) num_clusters="$2"; shift ;;
        --per_class_kmeans) per_class_kmeans="$2"; shift ;;
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

module purge
module load python/3.11
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${SLURM_TMPDIR}" || exit
echo "Current directory: ${SLURM_TMPDIR}"

rsync -a --exclude=".git" "${PROJECT_DIR}/" "${TMP_PROJECT_DIR}/"
cd "${TMP_PROJECT_DIR}" || exit
echo "Project copied to ${PWD}"

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cc.txt
#source /home/carson/DPTabula/tabpe/ENV/bin/activate

export PYTHONPATH="${TMP_PROJECT_DIR}:${PYTHONPATH}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

if [[ ("$generator_method" != "tabpe" || "$compare_method" != "tabpe")  && "$eval_only" == false ]]; then
    rsync -a "${MODEL_SRC}/" "${TMP_MODEL_DIR}/"
    echo "Model copied to ${TMP_MODEL_DIR}"
else
    echo "Skipping model copy (generator_method=$generator_method, compare_method=$compare_method)"
fi

OUTPUT_DIR="${RESULTS_DIR}/${SLURM_JOB_ID}/${dataset}/${seed}/pe/eps_${epsilon}_ns_${num_samples}_e_${epochs}_vm_${variance_multiplier}_dt_${decay_type}_gamma_${gamma}"
mkdir -p "$OUTPUT_DIR"

# Note that this copies the data to SCRATCH (so we save the split rather than just in TMPDIR)
echo "Running data split"
bash "${TMP_PROJECT_DIR}/scripts/data-split.sh" --dataset "$dataset" --seed "$seed"
echo "Data split completed"

# if [[ "$generator_method" != "tabpe" || "$compare_method" != "tabpe" ]]; then
#     echo "Running embedding"
#     bash "${TMP_PROJECT_DIR}/scripts/tabula_embeddings.sh" --dataset "$dataset" --seed "$seed"
#     echo "Data split completed"
# fi

mkdir -p "${TMP_DATA_DIR}/${dataset}"
rsync -a "${DATA_SRC}/" "${TMP_DATA_DIR}/${dataset}/"
echo "Data copied to ${TMP_DATA_DIR}"

SYNTH_DATA_DIR="/home/carson/scratch/logs/tabpe/${dataset}/epsilon_${epsilon}/seed_${seed}/compare_${compare_method}/generator_${generator_method}/per_class_kmeans_${per_class_kmeans}/num_clusters_${num_clusters}"
# ----- Run PE -----
if [[ "$eval_only" == false ]]; then
    echo -e "\nRunning PE ==============================================================="
    start_time=$(date +%s)
    start_time_readable=$(date)
    echo "Starting PE execution at: $start_time_readable"
    python -u "${TMP_PROJECT_DIR}/src/model/pe/rw_main_kmeans.py" \
        --epochs "$epochs" \
        --priv_train_csv "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}/data_train.csv" \
        --metadata_path "${TMP_DATA_DIR}/${dataset}/metadata.json" \
        --num_samples "$num_samples" \
        --variance_multiplier "$variance_multiplier" \
        --decay_type "$decay_type" \
        --gamma "$gamma" \
        --output_dir "${SYNTH_DATA_DIR}" \
        --epsilon "$epsilon" \
        --model_path "${TMP_MODEL_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --generator_method "${generator_method}" \
        --compare_method "${compare_method}" \
        --priv_train_emb "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}/data_train_emb.safetensors" \
        --seed "${seed}" \
        --num_clusters "${num_clusters}" \
        --per_class_kmeans "${per_class_kmeans}"

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
- Epsilon: $epsilon
- Seed: $seed
- generator_method: $generator_method
- compare_method: $compare_method
- num_clusters: $num_clusters
- per_class_kmeans: $per_class_kmeans
EOF

echo "Timing information saved to: ${OUTPUT_DIR}/timing.txt"
fi

# ----- Run evaluation -----
TMP_SYNTH_DIR="${SLURM_TMPDIR}/synthetic_data"
echo "Copying synthetic data to local scratch"
rm -rf "${TMP_SYNTH_DIR}"
mkdir -p "${TMP_SYNTH_DIR}"
rsync -a "${SYNTH_DATA_DIR}/" "${TMP_SYNTH_DIR}/"
echo "Synthetic data copied to ${TMP_SYNTH_DIR}"

echo -e "\nRunning Evaluation ==============================================================="
echo -e "Compare method using ${compare_method}"
python -u "${TMP_PROJECT_DIR}/src/evaluation/eval.py" \
    --epochs "$epochs" \
    --metadata_path "${TMP_DATA_DIR}/${dataset}/metadata.json" \
    --priv_train_csv "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}/data_train.csv" \
    --priv_val_csv "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}/data_val.csv" \
    --priv_test_csv "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}/data_test.csv" \
    --synthetic_data_dir "${TMP_SYNTH_DIR}" \
    --output_dir "${SYNTH_DATA_DIR}" \
    --classifier "$classifier"

echo -e "eval.py completed, now running eval_embedding.py ==============================================================="

python -u "${TMP_PROJECT_DIR}/src/evaluation/eval_embedding.py" \
    --dataset "$dataset" \
    --seed "$seed" \
    --metadata_path "${TMP_DATA_DIR}/${dataset}/metadata.json" \
    --priv_train_csv "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}/data_train.csv" \
    --priv_val_csv "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}/data_val.csv" \
    --priv_test_csv "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}/data_test.csv" \
    --synthetic_dir "${TMP_SYNTH_DIR}" \
    --output_dir "${SYNTH_DATA_DIR}"

echo "Done! Results saved to $OUTPUT_DIR"
