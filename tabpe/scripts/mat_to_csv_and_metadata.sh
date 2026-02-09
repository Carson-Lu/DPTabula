#!/bin/bash

# ===== Defaults =====
DATASET="BASEHOCK"
MAX_UNIQUE_RATIO=0.01
LABEL_COLUMN=""  # empty = default (last column)
BASE_PATH="/home/carson/scratch/data_tabpe"

# ===== Parse arguments =====
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data) DATASET="$2"; shift ;;
        --max_unique_ratio) MAX_UNIQUE_RATIO="$2"; shift ;;
        --label_column) LABEL_COLUMN="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Preprocessing dataset: $DATASET"

CSV_FILE="$BASE_PATH/$DATASET/$DATASET.csv"
METADATA_FILE="$BASE_PATH/$DATASET/metadata.json"

# Step 1: Convert .mat -> CSV
if [[ -f "$CSV_FILE" ]]; then
    echo "Step 1: CSV already exists at $CSV_FILE, skipping conversion."
else
    echo "Step 1: Converting .mat -> CSV"
    python /home/carson/DPTabula/tabpe/src/data_processing/mat_to_csv.py \
        --data "$DATASET" || { echo "mat_to_csv.py failed"; exit 1; }
fi


# Step 2: Generate metadata.json
echo "Step 2: Generating metadata.json"

CMD="python /home/carson/DPTabula/tabpe/src/data_processing/csv_to_metadata.py \
    --data $DATASET \
    --max_unique_ratio $MAX_UNIQUE_RATIO"

# Only pass label_column if it's not empty
if [[ -n "$LABEL_COLUMN" ]]; then
    CMD="$CMD --label_column $LABEL_COLUMN"
fi

echo "Running: $CMD"
$CMD || { echo "csv_to_metadata.py failed"; exit 1; }


echo "Preprocessing complete!"
echo "CSV: $CSV_FILE"
echo "Metadata: $METADATA_FILE"
