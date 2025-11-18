#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       # enough for data loading
#SBATCH --mem=32G               # adjust if your dataset is larger
#SBATCH --time=0:45:00
#SBATCH --gres=gpu:a100:1       # used only for embeddings
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-mijungp

cd DPTabula
module purge
module load python/3.13 scipy-stack


if [ ! -d ~/py313-cc ]; then
    python -m venv ~/py313-cc
fi

source ~/py313-cc/bin/activate
pip install --no-index -r requirements-cc.txt

python test_job.py