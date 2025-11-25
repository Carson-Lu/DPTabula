#!/bin/bash
#SBATCH --job-name=baseline-creditg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --gres=gpu:a100:1 
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/baseline_linear_classifier_%j.out
#SBATCH --error=/home/carson/scratch/logs/baseline_linear_classifier_%j.err

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
cp /home/carson/projects/rrg-mijungp/carson/data/pendigits.csv ./data/
echo "Data copied to $PWD/data/pendigits.csv"

# ----- Run baseline linear test -----
python -u ./baseline_linear_singledataset.py \
       --data_path ./data/pendigits.csv \
       --results_path /home/carson/scratch/Experiment Results/baseline_linear_classifier_results_pendigits_${SLURM_JOB_ID}.txt

echo "Done! Results saved to /home/carson/scratch/Experiment Results/baseline_linear_classifier_results_pendigits_${SLURM_JOB_ID}.txt"
