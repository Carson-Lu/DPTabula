#SBATCH --nodes=1
#SBATCH --mem-4G
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8             # match dataloader workers
#SBATCH --gpus-per-node=a100:1        # fill the node’s GPUs
#SBATCH --mem=64G

cd DPTabula
module purge
module load python/3.13 scipy-stack


if [ ! -d ~/py313-cc ]; then
    python -m venv ~/py313-cc
fi

source ~/py313-cc/bin/activate
pip install --no-index -r requirements-cc.txt

python test_job.py