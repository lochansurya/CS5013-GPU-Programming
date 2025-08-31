#!/bin/bash
echo "=========================="
echo "Problem-1_2: 2D Grid/Block multiplication"
echo "=========================="
echo "Running Problem-1_2 tests..."


# Ensure matmul_2d binary exists
BINARY="$SCRIPT_DIR/matmul_2d"
if [[ ! -x "$BINARY" ]]; then
    echo "Error: matmul_2d binary not found in $SCRIPT_DIR. Exiting."
    exit 1
fi

mkdir -p "$SCRIPT_DIR/results"

# Remove old concatenated file if it exists
CONCAT_FILE="$SCRIPT_DIR/results/results_1_2.txt"
rm -f "$CONCAT_FILE"

for MATRIX in "$SCRIPT_DIR"/public_test_cases/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    MATRIX_A="public_test_cases/matrix_${IDX}a.csv"
    MATRIX_B="public_test_cases/matrix_${IDX}b.csv"
    EXPECTED="$SCRIPT_DIR/output_matrix_mul.csv"
    OUTFILE="results/results_1_2_${IDX}.txt"

    echo "Matrix: $MATRIX" > "$OUTFILE"

    TRIAL_NO=1
    for X1 in 1 4 16; do
        for Y1 in 1 4 16; do
            for X2 in 16 32 64; do
                for Y2 in 16 32; do

                    # Skip invalid block sizes
                    if (( X2 * Y2 > 1024 )); then
                        echo "Skipping trial $TRIAL_NO: Invalid block size ($X2 x $Y2) = $((X2*Y2)) > 1024"
                        ((TRIAL_NO++))
                        continue
                    fi

                    echo "Running trial $TRIAL_NO: 2DGrid=($X1,$Y1) 2DBlock=($X2,$Y2)"

                    ./matmul_2d $X1 $Y1 $X2 $Y2 "$MATRIX_A" "$MATRIX_B" &>> "$OUTFILE"

                    # Extract kernel execution time (line 2)
                    TIME_MICRO=$(sed -n '2p' output_1_2_CS25MTECH11015.txt | awk '{print $NF}')

                    {
                        echo "============="
                        echo "Trial No. $TRIAL_NO"
                        echo "============="
                        echo "2DGrid(X1,Y1): ($X1,$Y1)"
                        echo "2DBlock(X2,Y2): ($X2,$Y2)"
                        echo "Kernel Execution Time: $TIME_MICRO microseconds"
                        echo "============="
                    } >> "$OUTFILE"

                    ((TRIAL_NO++))
                done
            done
        done
    done

    cat "$OUTFILE" >> "$CONCAT_FILE"
    echo -e "\n" >> "$CONCAT_FILE"
done

echo "All Problem-1_2 results saved to $CONCAT_FILE"
echo "Problem-1_2 tests completed."
echo "========================="
