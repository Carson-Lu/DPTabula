#!/bin/bash
#SBATCH --job-name=baseline-alldatasets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=def-mijungp
#SBATCH --output=/home/carson/scratch/logs/baseline_alldatasets_%j.out
#SBATCH --error=/home/carson/scratch/logs/baseline_alldatasets_%j.err

# ----- Define paths -----
PROJECT_DIR="/home/carson/DPTabula"
DATA_SOURCE="/home/carson/scratch/data"
RESULTS_DIR="/home/carson/scratch/Experiment_Results"
TMP_PROJECT_DIR="${SLURM_TMPDIR}/DPTabula"
TMP_DATA_DIR="${TMP_PROJECT_DIR}/data"

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
rsync -a "${DATA_SOURCE}/" "${TMP_DATA_DIR}/" || true
echo "Data copied to ${TMP_DATA_DIR}"

# ----- Run Python script on ALL datasets -----
python -u ./baseline_linear_alldatasets.py \
       --data_dir "${TMP_DATA_DIR}" \
       --results_path "${RESULTS_DIR}/baseline_alldatasets_results_${SLURM_JOB_ID}.txt"

# ----- Completion message -----
echo "Done! Results saved to ${RESULTS_DIR}/baseline_alldatasets_results_${SLURM_JOB_ID}.txt"
