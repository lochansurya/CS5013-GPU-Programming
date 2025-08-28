## Compiler
CC:= gcc
NVCC      := nvcc
CFLAGS    := -O2
SRC_HELPER:= matrix_csv.c
OBJ_HELPER:= matrix_csv.o

# Programs
PROGS := matmul_1_1 matmul_1_2 matmul_2 mattrans_3 mattrans_4

all: $(PROGS)
	@echo "All programs built successfully."

# Map program names to source files
matmul_1_1:   main_1_1.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

matmul_1_2:  main_1_2.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

matmul_2:    main_2.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

mattrans_3:    main_3.cu $(OBJ_HELPER)
	$(NVCC) $(CFLAGS) -o $@ $^

mattrans_4:    main_4.cu $(OBJ_HELPER)
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

