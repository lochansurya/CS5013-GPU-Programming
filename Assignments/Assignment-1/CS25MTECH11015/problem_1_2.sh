#!/bin/bash
echo "=========================="
echo "Problem-1_2: 2D Grid/Block multiplication"
echo "=========================="
echo "Running Problem-1_2 tests..."

# Generate matrices first if needed

mkdir -p results

CONCAT_FILE="results/results_1_2.txt"
rm -f "$CONCAT_FILE"

for MATRIX in matrices/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    MATRIX_A="matrices/matrix_${IDX}a.csv"
    MATRIX_B="matrices/matrix_${IDX}b.csv"
    EXPECTED="matrices/matrix_${IDX}c_expected.csv"
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

                    python3 tester.py "output_1_2_CS25MTECH11015.csv" "$EXPECTED"

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
