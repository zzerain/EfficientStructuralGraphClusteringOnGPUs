#pragma once


#include "Graph.h"


vid_t* GPUScan(graph_t* g, eid_t* num_edges_copy, vid_t* Edge_u, vid_t* Edge_v, double eps, int miu);

