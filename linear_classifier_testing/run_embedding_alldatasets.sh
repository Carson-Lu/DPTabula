#!/bin/bash
#SBATCH --job-name=LLM-alldatasets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:a100:1 
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/embedding_alldatasets_%j.out
#SBATCH --error=/home/carson/scratch/logs/embedding_alldatasets_%j.err

HOME="/home/carson"
PROJECT_SRC="${HOME}/DPTabula/linear_classifier_testing"
DATA_SRC="${HOME}/scratch/data"
RESULTS_DIR="${HOME}/scratch/Experiment_Results"

TMP_PROJECT_DIR="${SLURM_TMPDIR}/DPTabula/linear_classifier_testing"
TMP_DATA_DIR="${TMP_PROJECT_DIR}/data"
TMP_MODEL_DIR="${TMP_PROJECT_DIR}/model"

# ----- Default parameters -----
BATCH_SIZE=4
MODEL="tabula"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter: $1";
           echo "Usage: sbatch singledataset.sh --dataset <dataset_filename.csv> [--model tabula|llama] [--batch_size N]" ;
           exit 1 ;;
    esac
    shift
done

# ----- Model selection -----
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

cd "${SLURM_TMPDIR}" || exit
echo "Current directory: ${SLURM_TMPDIR}"

mkdir -p "${TMP_PROJECT_DIR}"
rsync -a --exclude=".git" "${PROJECT_SRC}/" "${TMP_PROJECT_DIR}/"
cd "${TMP_PROJECT_DIR}" || exit
echo "Project copied to ${PWD}"

module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip > /dev/null
pip install --no-index -r requirements-cc.txt > /dev/null

mkdir -p "${TMP_DATA_DIR}"
rsync -a "${DATA_SRC}/" "${TMP_DATA_DIR}/"
echo "Data copied to $TMP_DATA_DIR"

mkdir -p "${TMP_MODEL_DIR}"
rsync -a "${MODEL_SRC}/" "${TMP_MODEL_DIR}/"
echo "Model copied to ${TMP_MODEL_DIR}"

python -u ./embedding_linear_alldatasets.py \
    --data_dir "${TMP_DATA_DIR}" \
    --model_path "${TMP_MODEL_DIR}" \
    --results_path "${RESULTS_DIR}/${MODEL_NAME}_multidata_results_${SLURM_JOB_ID}.txt" \
    --batch_size "${BATCH_SIZE}"

echo "Done! Results saved to ${RESULTS_DIR}/${MODEL_NAME}_multidata_results_${SLURM_JOB_ID}.txt"