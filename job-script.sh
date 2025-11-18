#SBATCH --nodes=1
#SBATCH --mem-4G
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=ALL

python main.py