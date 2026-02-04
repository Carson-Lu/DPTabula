#!/bin/bash
#SBATCH --job-name=tabula_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1 
#SBATCH --mail-user=clu56@student.ubc.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --account=rrg-mijungp
#SBATCH --output=/home/carson/scratch/logs/tabula_embeddings_%j.out
#SBATCH --error=/home/carson/scratch/logs/tabula_embeddings_%j.err

# ----- Default parameters -----
dataset="adult"
seed=42
batch_size=4

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --model_path) batch_size="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

data_folder="/home/carson/scratch/data_tabpe/$dataset"
data_all="$data_folder/${dataset}.csv"

# ----- Run Tabula embedding script -----
python -u "${TMP_PROJECT_DIR}/src/data_processing/tabula_embeddings.py" \
    --data_dir "${TMP_DATA_DIR}/${dataset}/processed/${dataset}/${seed}" \
    --metadata_path "${TMP_DATA_DIR}/${dataset}/metadata.json" \
    --model_path "${TMP_MODEL_DIR}" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$batch_size"

echo "Tabula embeddings finished. Saved in $OUTPUT_DIR"
