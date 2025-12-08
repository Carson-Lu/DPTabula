#!/bin/bash
#SBATCH --job-name=baseline-singledataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=2G
#SBATCH --time=0:30:00
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=def-mijungp
#SBATCH --output=/home/carson/scratch/logs/baseline_singledataset_%j.out
#SBATCH --error=/home/carson/scratch/logs/baseline_singledataset_%j.err

HOME="/home/carson"
PROJECT_DIR="${HOME}/DPTabula/linear_classifier_testing"
DATA_SOURCE="${HOME}/scratch/data"
RESULTS_DIR="${HOME}/scratch/Experiment_Results"
TMP_PROJECT_DIR="${SLURM_TMPDIR}/linear_classifier_testing"
TMP_DATA_DIR="${TMP_PROJECT_DIR}/data"


# ----- Default parameters -----
dataset="pendigits.csv"   

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

cd "$SLURM_TMPDIR" || exit
echo "Current directory: $SLURM_TMPDIR"

rsync -a --exclude=".git" "${PROJECT_DIR}/" "${TMP_PROJECT_DIR}/"
cd "${TMP_PROJECT_DIR}" || exit
echo "Project copied to $PWD"

module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip > /dev/null
pip install --no-index -r requirements-cc.txt > /dev/null

mkdir -p "${TMP_DATA_DIR}"
rsync -a "${DATA_SOURCE}/" "${TMP_DATA_DIR}/"
echo "Data copied to $TMP_DATA_DIR"

python -u ./baseline_linear_singledataset.py \
       --data_path "${TMP_DATA_DIR}/${dataset}" \
       --results_path "${RESULTS_DIR}/baseline_singledataset_results_${SLURM_JOB_ID}.txt"
