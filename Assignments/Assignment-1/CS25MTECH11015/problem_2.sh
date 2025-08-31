#!/bin/bash
echo "=========================="
echo "Problem-2: Tiled Multiplication"
echo "=========================="
echo "Running Problem-2 tests..."

# Get the directory of this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

# Ensure matmul_tiled binary exists
BINARY="$SCRIPT_DIR/matmul_tiled"
if [[ ! -x "$BINARY" ]]; then
    echo "Error: matmul_tiled binary not found in $SCRIPT_DIR. Exiting."
    exit 1
fi

mkdir -p "$SCRIPT_DIR/results"

# Remove old concatenated file if it exists
CONCAT_FILE="$SCRIPT_DIR/results/results_2.txt"
rm -f "$CONCAT_FILE"

for MATRIX in "$SCRIPT_DIR"/public_test_cases/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    MATRIX_A="matrices/matrix_${IDX}a.csv"
    MATRIX_B="matrices/matrix_${IDX}b.csv"
    EXPECTED="$SCRIPT_DIR/output_matrix_mul.csv"
    OUTFILE="results/results_2_${IDX}.txt"

    echo "Matrix: $MATRIX" > "$OUTFILE"

    TRIAL_NO=1
    for TILE_WIDTH in 8 16 32 64 128; do
        echo "Running trial $TRIAL_NO: TileWidth=$TILE_WIDTH"

        ./matmul_tiled $TILE_WIDTH "$MATRIX_A" "$MATRIX_B" &>> "$OUTFILE"

        TIME_MICRO=$(sed -n '2p' output_2_CS25MTECH11015.txt | awk '{print $NF}')

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

echo "All Problem-2 results saved to $CONCAT_FILE"
echo "Problem-2 tests completed."
echo "========================="
