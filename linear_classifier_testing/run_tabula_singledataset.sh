#!/bin/bash
#SBATCH --job-name=tabula-singledataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1 
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/tabula_singledataset_%j.out
#SBATCH --error=/home/carson/scratch/logs/tabula_singledataset_%j.err

# ----- Check that a dataset filename was provided -----
if [ -z "$1" ]; then
    echo "Usage: sbatch tabula_singledataset.sh <dataset_filename.csv>"
    exit 1
fi

# ----- Define paths -----
PROJECT_DIR=/home/carson/DPTabula
DATA_DIR=/home/carson/scratch/data
DATA_SRC="${DATA_DIR}/$1"
MODEL_SRC=/home/carson/scratch/hf_models/tabula-8b
RESULTS_DIR=/home/carson/scratch/Experiment_Results
TMP_PROJECT_DIR=${SLURM_TMPDIR}/DPTabula
TMP_DATA_DIR=${TMP_PROJECT_DIR}/data
TMP_MODEL_DIR=${TMP_PROJECT_DIR}/tabula-8b
BATCH_SIZE=4

# Extract just the filename
DATA_FILENAME=$(basename "${DATA_SRC}")

# ----- Load Python environment -----
module purge
module load python/3.11
source ~/DPTabula/py311-cc/bin/activate

# ----- Move to fast local storage on node -----
cd "${SLURM_TMPDIR}" || exit
echo "Current directory: ${SLURM_TMPDIR}"

# ----- Clone repository (rsync instead of cp) -----
rsync -a --exclude=".git" "${PROJECT_DIR}/" "${TMP_PROJECT_DIR}/" || true
cd "${TMP_PROJECT_DIR}" || exit
echo "Project copied to ${PWD}"

# ----- Prepare data (rsync) -----
mkdir -p "${TMP_DATA_DIR}"
rsync -a "${DATA_SRC}" "${TMP_DATA_DIR}/" || true
echo "Data copied to ${TMP_DATA_DIR}"

# ----- Copy model to local storage (rsync) -----
rsync -a "${MODEL_SRC}/" "${TMP_MODEL_DIR}/" || true
echo "Model copied to ${TMP_MODEL_DIR}"

# ----- Run Python script -----
python -u ./tabula_linear_singledataset.py \
       --data_path "${TMP_DATA_DIR}/${DATA_FILENAME}" \
       --model_path "${TMP_MODEL_DIR}" \
       --results_path "${RESULTS_DIR}/tabula_singledataset_result_${SLURM_JOB_ID}.txt" \
       --batch_size ${BATCH_SIZE}

# ----- Completion message -----
echo "Done! Results saved to ${RESULTS_DIR}/tabula_singledataset_result_${SLURM_JOB_ID}.txt"
