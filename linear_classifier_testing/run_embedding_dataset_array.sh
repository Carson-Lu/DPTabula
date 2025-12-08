#!/bin/bash
#SBATCH --job-name=embedding_singledataset-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/embedding_singledataset_array_%A_%a.out
#SBATCH --error=/home/carson/scratch/logs/embedding_singledataset_array_%A_%a.err
#SBATCH --array=0-2

HOME="/home/carson"
PROJECT_SRC="${HOME}/DPTabula/linear_classifier_testing"
DATA_SRC="${HOME}/scratch/data"
RESULTS_DIR="${HOME}/scratch/Experiment_Results"

TMP_PROJECT_DIR="${SLURM_TMPDIR}/DPTabula/linear_classifier_testing"
TMP_DATA_DIR="${TMP_PROJECT_DIR}/data"
TMP_MODEL_DIR="${TMP_PROJECT_DIR}/model"

DATASETS=(
    "pc1.csv"
    "waveform-5000.csv"
    "wdbc.csv"
)

BATCH_SIZE=4
MODEL="tabula"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter: $1";
           echo "Usage: sbatch singledataset_array.sh [--model tabula|llama] [--batch_size N]" ;
           exit 1 ;;
    esac
    shift
done

if [[ "$MODEL" == "tabula" ]]; then
    MODEL_SRC="${HOME}/scratch/hf_models/tabula-8b"
    MODEL_NAME="tabula-8b"
elif [[ "$MODEL" == "llama" ]]; then
    MODEL_SRC="${HOME}/scratch/hf_models/llama-3-8B"
    MODEL_NAME="llama-3-8b"
else
    echo "Unknown model: $MODEL"
    exit 1
fi

echo "Model ${MODEL_NAME} selected."

DATA_FILENAME="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATA_FILE_SRC="${DATA_SRC}/${DATA_FILENAME}"
echo "Selected dataset: ${DATA_FILENAME}"

cd "${SLURM_TMPDIR}" || exit
echo "Current directory: ${SLURM_TMPDIR}"

mkdir -p "${TMP_PROJECT_DIR}"
rsync -a --exclude=".git" "${PROJECT_SRC}/" "${TMP_PROJECT_DIR}/"
cd "${TMP_PROJECT_DIR}" || exit
echo "Project copied to ${PWD}"

module load python/3.11
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip > /dev/null 2>&1
pip install --no-index -r requirements-cc.txt > /dev/null 2>&1

mkdir -p "${TMP_DATA_DIR}"
rsync -a "${DATA_FILE_SRC}" "${TMP_DATA_DIR}/"
echo "Data copied to ${TMP_DATA_DIR}"

mkdir -p "${TMP_MODEL_DIR}"
rsync -a "${MODEL_SRC}/" "${TMP_MODEL_DIR}/"
echo "Model copied to ${TMP_MODEL_DIR}"

python -u ./embedding_linear_singledataset.py \
    --data_path "${TMP_DATA_DIR}/${DATA_FILENAME}" \
    --model_path "${TMP_MODEL_DIR}" \
    --results_path "${RESULTS_DIR}/${MODEL_NAME}_singledataset_${DATA_FILENAME%.csv}_${SLURM_JOB_ID}.txt" \
    --batch_size "${BATCH_SIZE}"

echo "Finished processing ${DATA_FILENAME}"
