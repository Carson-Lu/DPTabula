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
source ~/DPTabula/py311-cc/bin/activate  # virtualenv with required packages

# ----- Move to fast local storage on node -----
cd $SLURM_TMPDIR
echo "Current directory: $SLURM_TMPDIR"

# ----- Clone repository -----
cp -r /home/carson/DPTabula ./DPTabula
cd DPTabula
echo "Project copied to $PWD"

# ----- Prepare data -----
mkdir -p data
cp /home/carson/projects/rrg-mijungp/carson/data/credit_g.csv ./data/
echo "Data copied to $PWD/data/credit_g.csv"

# ----- Copy model to local storage -----
cp -r /scratch/hf_models/tabula-8b ./tabula-8b
echo "Model copied to $PWD/tabula-8b"

# ----- Run Python script -----
python ./scripts/tabula_linear_test.py \
       --data_path ./data/credit_g.csv \
       --model_path ./tabula-8b \
       --results_path /scratch/tabula_creditg_results.txt

# ----- Completion message -----
echo "Done! Results saved to /scratch/tabula_creditg_results.txt"
