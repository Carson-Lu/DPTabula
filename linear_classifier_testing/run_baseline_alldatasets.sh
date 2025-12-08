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

HOME="/home/carson"
PROJECT_DIR="${HOME}/DPTabula/linear_classifier_testing"
DATA_SOURCE="${HOME}/scratch/data"
RESULTS_DIR="${HOME}/scratch/Experiment_Results"

TMP_PROJECT_DIR="${SLURM_TMPDIR}/linear_classifier_testing"
TMP_DATA_DIR="${TMP_PROJECT_DIR}/data"

cd "${SLURM_TMPDIR}" || exit
echo "Current directory: ${SLURM_TMPDIR}"

rsync -a --exclude=".git" "${PROJECT_DIR}/" "${TMP_PROJECT_DIR}/" || true
cd "${TMP_PROJECT_DIR}" || exit
echo "Project copied to ${PWD}"

module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip > /dev/null
pip install --no-index -r requirements-cc.txt > /dev/null

mkdir -p "${TMP_DATA_DIR}"
rsync -a "${DATA_SOURCE}/" "${TMP_DATA_DIR}/" || true
echo "Data copied to ${TMP_DATA_DIR}"

python -u ./baseline_linear_alldatasets.py \
       --data_dir "${TMP_DATA_DIR}" \
       --results_path "${RESULTS_DIR}/baseline_alldatasets_results_${SLURM_JOB_ID}.txt"

echo "Done! Results saved to ${RESULTS_DIR}/baseline_alldatasets_results_${SLURM_JOB_ID}.txt"
