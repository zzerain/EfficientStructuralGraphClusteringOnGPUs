#include <iostream>
#include <cstdio>
#include <climits>
#include <cassert>

#include <sys/time.h>

#include <chrono>
#include <sstream>
#include <fstream>

#include <omp.h>

//#include <set>

#include "GPUScan.h"

using namespace std::chrono;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "%s <Graph file> , parameter eps and miu\n", argv[0]);
        exit(1);
    }
 
    double eps = atof(argv[2]);
    int miu = atoi(argv[3]);
    graph_t g;

    //char file_dir[100];
    //cout << "input file directory" << endl;
    //cin >> file_dir;

    Graph scan_graph(argv[1]);
    g.adj = scan_graph.edge_dst;
    g.num_edges = scan_graph.node_off;
    g.n = scan_graph.nodemax;
    g.m = scan_graph.edgemax;

    int max_degree = 0;
    for (size_t i = 0; i < g.n; i++)
    {
        if (g.num_edges[i + 1] - g.num_edges[i] > max_degree)
        {
            max_degree = g.num_edges[i + 1] - g.num_edges[i];
        }
    }
    std:cout << "max degree: " << max_degree << "\n";

    auto gpuscan_start = high_resolution_clock::now();
    //get eid edgelist
    int* Edge_u = (vid_t*)malloc((g.m / 2) * sizeof(vid_t));
    int* Edge_v = (vid_t*)malloc((g.m / 2) * sizeof(vid_t));

    auto start = high_resolution_clock::now();
    getEidAndEdgeList(&g, Edge_u, Edge_v);
    auto end = high_resolution_clock::now();
    fprintf(stderr, "\ngetEidAndEdgeList time: %.3lf s\n", duration_cast<milliseconds>(end - start).count() / 1000.0);

    //cout<<"edge list:"<<endl;
    //for(int i=0;i<g.m/2;++i){
    //	cout<<Edge_u[i]<<" "<<Edge_v[i]<<endl;
    //}

    //cout<<"node:"<<endl;
    //for(int i=0;i<g.n+1;++i){
    //	cout<<g.num_edges[i]<<endl;
    //}
    //cout<<"adjcent:"<<endl;
    //for(int i=0;i<g.m;++i){
    //	cout<<g.adj[i]<<endl;
    //}
    fprintf(stderr, "node number and edge number: %ld, %ld\n", g.n, g.m);

    //GPUScan
    fprintf(stderr, "Start GPUScan\n");
    int* cluster = GPUScan(&g, Edge_u, Edge_v, eps, miu);
    fprintf(stderr, "Finish GPUScan\n");

    auto gpuscan_end = high_resolution_clock::now();
    fprintf(stderr, "gpuscan time: %.3lf s\n", duration_cast<milliseconds>(gpuscan_end - gpuscan_start).count() / 1000.0);

    bool* check_cluster = (bool*)malloc(g.n * sizeof(bool));
    int sum = 0;

#pragma omp parallel for
    for (auto i = 0; i < g.n; i++) {
        check_cluster[i] = false;
    }

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
    
    //std::set<int> s;
    //for (auto i = 0; i < g.n; ++i) {
    //    s.insert(cluster[i]);
    //}
    //fprintf(stderr, "Cluster Number: %d\n", s.size());

    fprintf(stderr, "Cluster Number: %d\n", sum);

    //free
    free_graph(&g);
    free(Edge_u);
    free(Edge_v);
    free(check_cluster);
    free(cluster);

    fprintf(stderr, "Work Complete!\n");

    return 0;
}
