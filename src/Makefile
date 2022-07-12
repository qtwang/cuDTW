NVCC= nvcc
NVCCFLAGS= -O3 -std=c++14 -arch=sm_70
CXXFLAGS= -Xcompiler="-fopenmp -march=native"

all: main pairwise

main: main.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) main.cu -o main

pairwise: pairwise.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) pairwise.cu -o pairwise

clean:
	rm -f main pairwise
