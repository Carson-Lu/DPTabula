#!/bin/bash
#SBATCH --job-name=tabula-singledataset-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/tabula_singledataset_%A_%a.out
#SBATCH --error=/home/carson/scratch/logs/tabula_singledataset_%A_%a.err
#SBATCH --array=0-0   # <-- will update automatically below based on dataset count

# ----- Python environment -----
module purge
module load python/3.11
source ~/DPTabula/py311-cc/bin/activate

# ----- Define paths -----
PROJECT_DIR=/home/carson/DPTabula
DATA_DIR=/home/carson/scratch/data
MODEL_SRC=/home/carson/scratch/hf_models/tabula-8b
RESULTS_DIR=/home/carson/scratch/Experiment_Results
TMP_PROJECT_DIR=${SLURM_TMPDIR}/DPTabula
TMP_DATA_DIR=${TMP_PROJECT_DIR}/data
TMP_MODEL_DIR=${TMP_PROJECT_DIR}/tabula-8b
BATCH_SIZE=16

# ----- List of datasets -----
DATASETS=(
    "madelon.csv"
    "mfeat-fourier.csv"
    "pc1.csv"
    "pc3.csv"
    "pc4.csv"
    "pendigits.csv"
    "phoneme.csv"
    "poker-hand.csv"
    "qsar-biodeg.csv"
    "waveform-5000.csv"
    "wdbc.csv"
)

# ----- Set SLURM array based on list length -----
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "This script should be submitted as a SLURM array job."
    exit 1
fi

# ----- Select dataset for this array task -----
DATA_FILENAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}
DATA_SRC="${DATA_DIR}/${DATA_FILENAME}"

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
python -u "${TMP_PROJECT_DIR}/tabula_linear_singledataset.py" \
       --data_path "${TMP_DATA_DIR}/${DATA_FILENAME}" \
       --model_path "${TMP_MODEL_DIR}" \
       --results_path "${RESULTS_DIR}/tabula_singledataset_result_${DATA_FILENAME%.csv}_${SLURM_JOB_ID}.txt" \
       --batch_size ${BATCH_SIZE}

echo "Finished processing ${DATA_FILENAME}"
