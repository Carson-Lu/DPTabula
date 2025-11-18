#SBATCH --nodes=1
#SBATCH --mem-4G
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8             # match dataloader workers
#SBATCH --gpus-per-node=a100:1        # fill the node’s GPUs
#SBATCH --mem=64G


python test_job.py