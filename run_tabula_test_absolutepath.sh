#!/bin/bash
#SBATCH --job-name=tabula-creditg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # enough for data loading
#SBATCH --mem=16G                   # adjust if your dataset is larger
#SBATCH --time=0:45:00
#SBATCH --gres=gpu:a100:1           # GPU for embeddings
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-mijungp
#SBATCH --output=tabula_creditg_%j.out
#SBATCH --error=tabula_creditg_%j.err

# ----- Load Python environment -----
module purge
module load python/3.11
source /home/carson/DPTabula/py311/bin/activate  # virtualenv with required packages

# ----- Move to project directory -----
cd /home/carson/DPTabula

# ----- Run Python script with absolute paths -----
python ./DPTabula/tabula_linear_test.py \
       --data_path /home/carson/projects/rrg-mijungp/carson/data/credit_g.csv \
       --model_path /home/carson/scratch/hf_models/tabula-8b \
       --results_path /home/carson/scratch/tabula_creditg_results.txt

echo "Done! Results saved to /scratch/tabula_creditg_results.txt"
