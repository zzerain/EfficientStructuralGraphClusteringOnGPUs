#include <iostream>
#include <cstdio>
#include <climits>
#include <cassert>

#include <chrono>
#include <sstream>
#include <fstream>

#include <omp.h>

#include "SCANum.h"

int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "%s <Graph file> , parameter eps and miu\n", argv[0]);
        exit(1);
    }

    double eps = atof(argv[2]);
    int miu = atoi(argv[3]);
    
    graph_t g;

    Graph scan_graph(argv[1]);
    g.adj = scan_graph.edge_dst;
    g.num_edges = scan_graph.node_off;
    g.n = scan_graph.nodemax;
    g.m = scan_graph.edgemax;
    scan_graph.degree.clear();

    //vector<vid_t> new_vid_dict;
    //vector<vid_t> old_vid_dict;
    //ReorderDegAscending(g, new_vid_dict, old_vid_dict);

    using namespace std::chrono;

    //get eid edgelist
    vid_t* Edge_u = (vid_t*)malloc((g.m / 2) * sizeof(vid_t));
    vid_t* Edge_v = (vid_t*)malloc((g.m / 2) * sizeof(vid_t));
    eid_t* num_edges_copy = (eid_t*)malloc((g.n + 1) * sizeof(eid_t));
    getEidAndEdgeList(&g, Edge_u, Edge_v, num_edges_copy);
    free(num_edges_copy);
    auto start = high_resolution_clock::now();
    // GPUScan
    vid_t* cluster = SCANum(&g, Edge_u, Edge_v, eps, miu);

    auto end = high_resolution_clock::now();
    fprintf(stderr, "gpuscan time: %.3lf s\n", duration_cast<milliseconds>(end - start).count() / 1000.0);

    bool* check_cluster = (bool*)malloc(g.n * sizeof(bool));
    for (auto i = 0; i < g.n; i++) {
        check_cluster[i] = false;
    }
    int sum = 0;
#pragma omp parallel
    {
#pragma omp for
        for (auto i = 0; i < g.n; ++i) {
            if (cluster[i] >= 0) check_cluster[cluster[i]] = true;
        }
#pragma omp single
        {
            for (auto i = 0; i < g.n; i++) {
                if (check_cluster[i]) sum++;
            }
        }
    }

    fprintf(stderr, "Cluster Number: %d\n", sum);

    //free
    free_graph(&g);
    free(Edge_u);
    free(Edge_v);
    free(cluster);
    free(check_cluster);

    fprintf(stderr, "finish\n");

    return 0;
}

