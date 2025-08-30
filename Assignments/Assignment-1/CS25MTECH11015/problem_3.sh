#!/bin/bash
echo "=========================="
echo "Problem-3: Basic Transpose"
echo "=========================="
echo "Running Problem-3 tests..."


mkdir -p results

CONCAT_FILE="results/results_3.txt"
rm -f "$CONCAT_FILE"

for MATRIX in matrices/transpose/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    EXPECTED="matrices/transpose/matrix_${IDX}a_T.csv"
    OUTFILE="results/results_3_${IDX}.txt"

    echo "Matrix: $MATRIX" > "$OUTFILE"

    TRIAL_NO=1
    for M in 64 128 256; do
        for N in 64 128 256; do
            echo "Running trial $TRIAL_NO: Size=($M,$N)"

            ./mattrans_basic $M $N "$MATRIX" &>> "$OUTFILE"

            TIME_MICRO=$(sed -n '2p' output_3_CS25MTECH11015.txt | awk '{print $NF}')

            {
                echo "============="
                echo "Trial No. $TRIAL_NO"
                echo "Matrix Size: ($M,$N)"
                echo "Kernel Execution Time: $TIME_MICRO microseconds"
                echo "============="
            } >> "$OUTFILE"

            python3 tester.py "output_3_CS25MTECH11015.csv" "$EXPECTED"

            ((TRIAL_NO++))
        done
    done

    cat "$OUTFILE" >> "$CONCAT_FILE"
    echo -e "\n" >> "$CONCAT_FILE"
done

echo "All Problem-3 results saved to $CONCAT_FILE"
echo "Problem-3 tests completed."
echo "========================="
