#!/usr/bin/env bash
echo "=========================="
echo "Problem-3: Matrix Transpose 1D"
echo "=========================="
echo "Running Problem-3 tests..."

# Get the directory of this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

OUTFILE="$SCRIPT_DIR/results/results_3.txt"
# Ensure mattrans_basic binary exists
BINARY="$SCRIPT_DIR/mattrans_basic"
if [[ ! -x "$BINARY" ]]; then
    echo "Error: mattrans_basic binary not found in $SCRIPT_DIR. Exiting."
    exit 1
fi

mkdir -p "$SCRIPT_DIR/results"

# Remove old concatenated file if it exists
CONCAT_FILE="$SCRIPT_DIR/results/results_3.txt"
rm -f "$CONCAT_FILE"

EXPECTED="$SCRIPT_DIR/output_matrix_transpose.csv"

TRIAL_NO=1
for M in 2 4 8 16 32 64; do
    for N in 2 4 8 16 32 64; do
        # Skip if total threads exceed 2048
        TOTAL_THREADS=$((M * N))
        if [ "$TOTAL_THREADS" -gt 2048 ]; then
            echo "Skipping trial $TRIAL_NO: total threads ($TOTAL_THREADS) exceed 2048"
            ((TRIAL_NO++))
            continue
        fi

        echo "Running trial $TRIAL_NO: Size=($M,$N)"

        # Run binary and capture stdout
        OUTPUT=$(./mattrans_basic $M $N "$SCRIPT_DIR/public_test_cases/matrix_a.csv")
        echo "$OUTPUT" >> "$OUTFILE"

        # Extract time from captured output
        TIME_MICRO=$(echo "$OUTPUT" | grep "Kernel execution time" | awk '{print $(NF-1)}')

        {
            echo "============="
            echo "Trial No. $TRIAL_NO"
            echo "grid_x, block_x: ($M,$N)"
            echo "Kernel Execution Time: $TIME_MICRO microseconds"
            echo "============="
        } >> "$OUTFILE"

        ((TRIAL_NO++))
    done
done
