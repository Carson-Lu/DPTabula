#!/bin/bash
#SBATCH --job-name=tabula-alldatasets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1 
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
cp /home/carson/projects/rrg-mijungp/carson/data/*.csv ./data/
echo "Data copied to $PWD/data/"

on/hf_models/tabula-8b ./tabula-8b
echo "Model copied to $PWD/tabula-8b"

# Run Python script on ALL datasets
python -u ./tabula_linear_alldatasets.py \
    --data_dir ./data \
    --results_path /home/carson/scratch/Experiment_Results/baseline_alldatasets_results_${SLURM_JOB_ID}.txt

# ----- Completion message -----
echo "Done! Results saved to /home/carson/scratch/Experiment_Results/baseline_alldatasets_results_${SLURM_JOB_ID}.txt"
