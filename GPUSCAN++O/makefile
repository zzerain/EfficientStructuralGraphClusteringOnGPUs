MACRO=-D_DEBUG_
CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp $(MACRO) -g
NVCC=nvcc
NVCCFLAGS=--gpu-architecture=compute_60 --gpu-code=sm_60 -Xcompiler -O3 -std=c++11 $(MACRO) -g -G

SCANoffload : main.o Graph.o SCANoffload.o gpu.o
	$(CXX) $(CXXFLAGS) -o SCANoffload main.o Graph.o SCANoffload.o gpu.o -L /usr/local/cuda-10.1/lib64 -lcudart -lcudadevrt

Graph.o : Graph.cpp
	$(CXX) $(CXXFLAGS) -c Graph.cpp

SCANoffload.o : SCANoffload.cu
	$(NVCC) $(NVCCFLAGS) -dc SCANoffload.cu -o SCANoffload.o

gpu.o : SCANoffload.o
	$(NVCC) $(NVCCFLAGS) -dlink  SCANoffload.o -o gpu.o

main.o : main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

.PHONY : clean
clean :
	rm SCANoffload  *.o
