#pragma once


//#include "cuda_runtime.h"
#include "Graph.h"


int* GPUScan(graph_t* g, int* Edge_u, int* Edge_v, double eps, int miu);
