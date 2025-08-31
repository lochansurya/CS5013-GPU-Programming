TRIAL_NO=1
for M in 64 128 256; do
    for N in 64 128 256; do
        # Skip if total threads exceed 2048
        TOTAL_THREADS=$((M * N))
        if [ "$TOTAL_THREADS" -gt 2048 ]; then
            echo "Skipping trial $TRIAL_NO: total threads ($TOTAL_THREADS) exceed 2048"
            ((TRIAL_NO++))
            continue
        fi

        echo "Running trial $TRIAL_NO: Size=($M,$N)"

        ./mattrans_basic $M $N "$MATRIX" &>> "$OUTFILE"

        TIME_MICRO=$(sed -n '2p' output_3_CS25MTECH11015.txt | awk '{print $NF}')

        {
            echo "============="
            echo "Trial No. $TRIAL_NO"
            echo "grid_x, block_x: ($M,$N)"
            echo "Kernel Execution Time: $TIME_MICRO microseconds"
            echo "============="
        } >> "$OUTFILE"

        python3 tester.py "public_test_cases/output_3_CS25MTECH11015.csv" "$EXPECTED"

        ((TRIAL_NO++))
    done
done
