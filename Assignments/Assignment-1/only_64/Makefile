## Compiler
CC:= gcc
NVCC      := nvcc
CFLAGS    := -O2
SRC_HELPER:= matrix_csv.c
OBJ_HELPER:= matrix_csv.o
BIN_DIR := bin

#
# Programs
PROGS := matmul_1d matmul_2d matmul_tiled mattrans_basic mattrans_tiled

all: $(PROGS)
	@echo "All programs built successfully."

# Map program names to source files
matmul_1d:   main_1_1.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

matmul_2d:  main_1_2.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

matmul_tiled:    main_2.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

mattrans_basic:    main_3.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

mattrans_tiled:    main_4.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

# Object rule for helper C file
$(OBJ_HELPER): $(SRC_HELPER) matrix_csv.h
	$(CC) -c $(SRC_HELPER) -o $@

# Default target: build all programs
all: $(PROGS)

# Clean
clean:
	@echo "Removing all generated binaries..."
	rm -f $(PROGS) $(OBJ_HELPER)

