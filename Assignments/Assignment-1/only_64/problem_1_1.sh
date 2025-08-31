#!/bin/bash
set -euo pipefail  # Exit on error, undefined variable, or any pipeline failure

# Get the directory of this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

echo "=========================="
echo "Problem-1_1: 1D Grid/Block multiplication"
echo "=========================="

# Ensure matmul_1d binary exists
BINARY="$SCRIPT_DIR/matmul_1d"
if [[ ! -x "$BINARY" ]]; then
    echo "Error: matmul_1d binary not found in $SCRIPT_DIR. Exiting."
    exit 1
fi

mkdir -p "$SCRIPT_DIR/results"

# Remove old concatenated file if it exists
CONCAT_FILE="$SCRIPT_DIR/results/results_1_1.txt"
rm -f "$CONCAT_FILE"

# Loop through input matrices
for MATRIX in "$SCRIPT_DIR"/matrices/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}  # extract number
    MATRIX_A="$SCRIPT_DIR/public_test_cases/matrix_$a.csv"
    MATRIX_B="$SCRIPT_DIR/public_test_cases/matrix_$b.csv"
    EXPECTED="$SCRIPT_DIR/output_matrix_mul.csv"
    OUTFILE="$SCRIPT_DIR/results/results_1_1_${IDX}.txt"

    echo "Matrix: $MATRIX" > "$OUTFILE"

    TRIAL_NO=1
    for M in 16 32 64; do
        for N in 16 32 64; do
            echo "Running trial $TRIAL_NO: GridBlocks=$M ThreadsPerBlock=$N"

            # Run CUDA kernel; exit if it fails
            if ! "$BINARY" "$M" "$N" "$MATRIX_A" "$MATRIX_B" &>> "$OUTFILE"; then
                echo "Error: CUDA binary ./matmul_1d failed! Exiting." >> "$OUTFILE"
                exit 1
            fi

            # Check if the output txt file exists
            TXT_FILE="$SCRIPT_DIR/output_1_1_CS25MTECH11015.txt"
            if [[ ! -f "$TXT_FILE" ]]; then
                echo "Error: $TXT_FILE not found! Exiting." >> "$OUTFILE"
                exit 1
            fi

            # Extract kernel execution time from line 2
            TIME_MICRO=$(sed -n '2p' "$TXT_FILE" | awk '{print $NF}')

            # Append trial info
            {
                echo "============="
                echo "Trial No. $TRIAL_NO"
                echo "============="
                echo "#Thread Blocks(M): $M"
                echo "#Threads Per Block(N): $N"
                echo "Kernel Execution Time: $TIME_MICRO microseconds"
                echo "============="
            } >> "$OUTFILE"

            # Run tester for correctness (outputs to terminal only)
            python3 "$SCRIPT_DIR/tester.py" "$SCRIPT_DIR/public_test_cases/output_1_1_CS25MTECH11015.csv" "$EXPECTED"

            ((TRIAL_NO++))
        done
    done

    # Append this matrix's results to the final concatenated file
    cat "$OUTFILE" >> "$CONCAT_FILE"
    echo -e "\n" >> "$CONCAT_FILE"
done

echo "All Problem-1_1 results saved to $CONCAT_FILE"
echo "Problem-1_1 tests completed."
echo "========================="
