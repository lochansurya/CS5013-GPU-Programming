#!/usr/bin/env bash

# ==========================
# 0. Clean previous outputs
# ==========================
rm -f output_*.csv

# ==========================
# 1. Build all binaries
# ==========================
echo "Building all binaries..."
make all
echo "Build completed."
echo "-------------------------"

# ==========================
# 2. Generate matrices
# ==========================
echo "Generating matrices..."
python3 matrices_generator.py
python3 transpose_generator.py
echo "Matrices generated."
echo "-------------------------"

# ==========================
# 3. Problem-1_1: 1D Grid/Block multiplication
# ==========================
echo ==========================
echo "Problem-1_1: 1D Grid/Block multiplication"
echo  "=========================="
echo "Running Problem-1_1 tests..."
bash problem_1_1.sh
echo " "
echo "Problem-1_1 tests completed."
echo "========================="

# ==========================
# 4. Problem-1_2: 2D Grid/Block multiplication
# ==========================
echo ==========================
echo "Problem-1_2: 2D Grid/Block multiplication"
echo  "=========================="
echo "Running Problem-1_2 tests..."
for MATRIX in matrices/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    MATRIX_A="matrices/matrix_${IDX}a.csv"
    MATRIX_B="matrices/matrix_${IDX}b.csv"
    EXPECTED="matrices/matrix_${IDX}c_expected.csv"

    for X1 in 8 16; do
        for Y1 in 8 16; do
            for X2 in 8 16; do
                for Y2 in 8 16; do
                    echo "Matrix=$MATRIX 2DGrid=($X1,$Y1) 2DBlock=($X2,$Y2)"
                    ./matmul_2d $X1 $Y1 $X2 $Y2 "$MATRIX_A" "$MATRIX_B"
                    python3 tester.py "output_1_2_CS25MTECH11015.csv" "$EXPECTED"
                    echo "-------------------------------------------"
                done
            done
        done
    done
done
echo "Problem-1_2 tests completed."
echo "========================="

# ==========================
# 5. Problem-2: Tiled Multiplication
# ==========================
echo ==========================
echo "Problem-2: Tiled Multiplication"
echo  "=========================="
echo "Running Problem-2 tests..."
for MATRIX in matrices/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    MATRIX_A="matrices/matrix_${IDX}a.csv"
    MATRIX_B="matrices/matrix_${IDX}b.csv"
    EXPECTED="matrices/matrix_${IDX}c_expected.csv"

    for TILE_WIDTH in 8 16 32; do
        echo "Matrix=$MATRIX TileWidth=$TILE_WIDTH"
        ./matmul_tiled $TILE_WIDTH "$MATRIX_A" "$MATRIX_B"
        python3 tester.py "output_2_CS25MTECH11015.csv" "$EXPECTED"
        echo "-------------------------------------------"
    done
done
echo "Problem-2 tests completed."
echo "========================="

# ==========================
# 6. Problem-3: Basic Transpose
# ==========================
echo ==========================
echo "Problem-3: Basic Transpose"
echo  "=========================="
echo "Running Problem-3 tests..."
for MATRIX in matrices/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    EXPECTED="matrices/transpose/matrix_${IDX}a_T.csv"

    for M in 128 256; do
        for N in 128 256; do
            echo "Matrix=$MATRIX Size=($M,$N)"
            ./mattrans_basic $M $N "$MATRIX"
            python3 tester.py "output_3_CS25MTECH11015.csv" "$EXPECTED"
            echo "-------------------------------------------"
        done
    done
done
echo "Problem-3 tests completed."
echo "========================="

# ==========================
# 7. Problem-4: Tiled Transpose
# ==========================
echo ==========================
echo "Problem-4: Tiled Transpose"
echo  "=========================="
echo "Running Problem-4 tests..."
for MATRIX in matrices/matrix_*a.csv; do
    BASENAME=$(basename "$MATRIX" .csv)
    IDX=${BASENAME:7:1}
    EXPECTED="matrices/transpose/matrix_${IDX}a_T.csv"

    for TILE_WIDTH in 8 16 32; do
        echo "Matrix=$MATRIX TileWidth=$TILE_WIDTH"
        ./mattrans_tiled $TILE_WIDTH "$MATRIX"
        python3 tester.py "output_4_CS25MTECH11015.csv" "$EXPECTED"
        echo "-------------------------------------------"
    done
done
echo "Problem-4 tests completed."
echo "========================="

# ==========================
# 8. Clean all binaries
# ==========================
make clean
echo "All done."
