#!/bin/bash
#SBATCH --job-name=tabula_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1 
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/tabula_embeddings_%j.out
#SBATCH --error=/home/carson/scratch/logs/tabula_embeddings_%j.err

# ----- Default parameters -----
dataset="adult"
seed=42
batch_size=4

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
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

export PYTHONPATH="${TMP_PROJECT_DIR}:${PYTHONPATH}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Copy model for Tabula embeddings
rsync -a "${MODEL_SRC}/" "${TMP_MODEL_DIR}/"
echo "Model copied to ${TMP_MODEL_DIR}"

# Make output dir
mkdir -p "${TMP_OUTPUT_DIR}"
OUTPUT_DIR="${RESULTS_DIR}/${SLURM_JOB_ID}/${dataset}/${seed}/embeddings"
mkdir -p "$OUTPUT_DIR"

# Copy data to TMPDIR for faster access
mkdir -p "${TMP_DATA_DIR}/${dataset}"
rsync -a "${DATA_SRC}/" "${TMP_DATA_DIR}/${dataset}/"
echo "Data copied to ${TMP_DATA_DIR}"

# ----- Run Tabula embedding script -----
python -u "${TMP_PROJECT_DIR}/src/data_processing/tabula_embeddings.py" \
    --data_dir "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}" \
    --metadata_path "${TMP_DATA_DIR}/${dataset}/metadata.json" \
    --model_path "${TMP_MODEL_DIR}" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$batch_size"

echo "Tabula embeddings finished. Saved in $OUTPUT_DIR"
