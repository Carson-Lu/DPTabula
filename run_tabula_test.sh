#!/bin/bash
#SBATCH --job-name=tabula-creditg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # enough for data loading
#SBATCH --mem=64G                   # adjust if your dataset is larger
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1           # GPU for embeddings
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/tabula_linear_classifier%j.out
#SBATCH --error=/home/carson/scratch/logs/tabula_linear_classifier%j.err

# ----- Load Python environment -----
module purge
module load python/3.11
source ~/DPTabula/py311-cc/bin/activate

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
cp -r /home/carson/projects/rrg-mijungp/carson/hf_models/tabula-8b ./tabula-8b
echo "Model copied to $PWD/tabula-8b"

# ----- Run Python script -----
python -u ./tabula_linear_test.py \
       --data_path ./data/credit_g.csv \
       --model_path ./tabula-8b \
       --results_path /home/carson/scratch/logs/tabula_linear_classifier_results_${SLURM_JOB_ID}.out.txt

# ----- Completion message -----
echo "Done! Results saved to /home/carson/scratch/logs/tabula_linear_classifier${SLURM_JOB_ID}.txt"
