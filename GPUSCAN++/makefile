MACRO=#-D_DEBUG_
CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp -pthread $(MACRO) -g
NVCC=nvcc
NVCCFLAGS=--gpu-architecture=compute_70 --gpu-code=sm_70 -Xcompiler -O3 -std=c++11 $(MACRO) -g -G

GPUScan : main.o Graph.o GPUScan.o gpu.o
	$(CXX) $(CXXFLAGS) -o GPUScan main.o Graph.o GPUScan.o gpu.o -L /home/mycuda/cuda-10.1/lib64 -lcudart -lcudadevrt

Graph.o : Graph.cpp
	$(CXX) $(CXXFLAGS) -c Graph.cpp

GPUScan.o : GPUScan.cu
	$(NVCC) $(NVCCFLAGS) -dc GPUScan.cu -o GPUScan.o

gpu.o : GPUScan.o
	$(NVCC) $(NVCCFLAGS) -dlink  GPUScan.o -o gpu.o

main.o : main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

.PHONY : clean
clean :
	rm GPUScan  *.o
