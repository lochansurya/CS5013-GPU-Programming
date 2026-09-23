main: src/main.cu src/parANN.cu src/parANN.cuh
	nvcc src/main.cu src/parANN.cu -Xcompiler -fopenmp -std=c++20 -I./src/utils  -o build/bang_exact -O3 -g
 

clean:
	rm -f build/bang_exact 


