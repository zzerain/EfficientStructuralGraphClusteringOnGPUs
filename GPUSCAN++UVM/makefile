MACRO=-D_DEBUG_
CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp $(MACRO) -g
NVCC=nvcc
NVCCFLAGS=--gpu-architecture=compute_60 --gpu-code=sm_60 -Xcompiler -O3 -std=c++11 $(MACRO) -g -G

SCANum : main.o Graph.o SCANum.o gpu.o
	$(CXX) $(CXXFLAGS) -o SCANum main.o Graph.o SCANum.o gpu.o -L /usr/local/cuda-10.1/lib64 -lcudart -lcudadevrt

Graph.o : Graph.cpp
	$(CXX) $(CXXFLAGS) -c Graph.cpp

SCANum.o : SCANum.cu
	$(NVCC) $(NVCCFLAGS) -dc SCANum.cu -o SCANum.o

gpu.o : SCANum.o
	$(NVCC) $(NVCCFLAGS) -dlink  SCANum.o -o gpu.o

main.o : main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

.PHONY : clean
clean :
	rm SCANum  *.o
