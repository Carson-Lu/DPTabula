#!/bin/bash
#SBATCH --job-name=tabula_test
#SBATCH --account=rrg-mijungp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/%u/logs/tabula_test_%j.out
#SBATCH --error=/scratch/%u/logs/tabula_test_%j.err

set -euo pipefail

# Paths
REPO_HOME=/home/carson/DPTabula
MODEL_HOME=/home/carson/scratch/hf_models/tabula-8b

# Stage repo and model to local tmp for fast I/O
TMP_REPO="${SLURM_TMPDIR}/DPTabula"
TMP_MODEL="${SLURM_TMPDIR}/hf_models/tabula-8b"

mkdir -p "${TMP_REPO}" "${TMP_MODEL}"
rsync -a --exclude=".git" "${REPO_HOME}/" "${TMP_REPO}/"
rsync -a "${MODEL_HOME}/" "${TMP_MODEL}/"

cd "${TMP_REPO}"

# Virtual environment
TMP_VENV="${SLURM_TMPDIR}/venv"
python -m venv "${TMP_VENV}"
source "${TMP_VENV}/bin/activate"

pip install --upgrade pip
pip install torch transformers --no-cache-dir

# Set Hugging Face caches to tmp for offline use
export HF_HOME="${SLURM_TMPDIR}/hf_home"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets"
export HF_MODELS_CACHE="${TMP_MODEL}"
export TRANSFORMERS_CACHE="${TMP_MODEL}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_MODELS_CACHE}"

# Run your main.py
python main.py --hf_model_path "${TMP_MODEL}"
