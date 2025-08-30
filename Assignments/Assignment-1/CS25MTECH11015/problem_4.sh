#!/bin/bash
echo "=========================="
echo "Problem-4: Tiled Transpose"
echo "=========================="
echo "Running Problem-4 tests..."

# Get the directory of this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

EXPECTED="$SCRIPT_DIR/output_matrix_transpose.csv"

# Ensure matmul_tiled binary exists
BINARY="$SCRIPT_DIR/mattrans_tiled"
if [[ ! -x "$BINARY" ]]; then
    echo "Error: mattrans_tiled binary not found in $SCRIPT_DIR. Exiting."
    exit 1
fi

mkdir -p "$SCRIPT_DIR/results"

# Remove old concatenated file if it exists
CONCAT_FILE="$SCRIPT_DIR/results/results_4.txt"
rm -f "$CONCAT_FILE"

for MATRIX in public_test_cases/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    EXPECTED="matrices/transpose/matrix_${IDX}a_T.csv"
    OUTFILE="results/results_4_${IDX}.txt"

    echo "Matrix: $MATRIX" > "$OUTFILE"

    TRIAL_NO=1
    for TILE_WIDTH in 2 4 8 16 32 64 128 256; do

        # Skip if TILE_WIDTH squared exceeds 2048
        if (( TILE_WIDTH * TILE_WIDTH > 2048 )); then
            echo "Skipping trial $TRIAL_NO: TileWidth=$TILE_WIDTH (exceeds limit)"
            ((TRIAL_NO++))
            continue
        fi

        echo "Running trial $TRIAL_NO: TileWidth=$TILE_WIDTH"

        ./mattrans_tiled $TILE_WIDTH "$MATRIX" &>> "$OUTFILE"

        TIME_MICRO=$(sed -n '2p' output_4_CS25MTECH11015.txt | awk '{print $NF}')

        {
            echo "============="
            echo "Trial No. $TRIAL_NO"
            echo "TileWidth: $TILE_WIDTH"
            echo "Kernel Execution Time: $TIME_MICRO microseconds"
            echo "============="
        } >> "$OUTFILE"

        ((TRIAL_NO++))
    done

    cat "$OUTFILE" >> "$CONCAT_FILE"
    echo -e "\n" >> "$CONCAT_FILE"
done

echo "All Problem-4 results saved to $CONCAT_FILE"
echo "Problem-4 tests completed."
echo "========================="
