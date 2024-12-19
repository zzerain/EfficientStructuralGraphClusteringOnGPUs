#include "Graph.h"

#include <cassert>

#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <omp.h>

#include "search_util.h"




inline int FindSrc(graph_t* g, int u, uint32_t edge_idx) {
    if (edge_idx >= g->num_edges[u + 1]) {
        // update last_u, preferring galloping instead of binary search because not large range here
        u = GallopingSearch(g->num_edges, static_cast<uint32_t>(u) + 1, g->n + 1, edge_idx);
        // 1) first > , 2) has neighbor
        if (g->num_edges[u] > edge_idx) {
            while (g->num_edges[u] - g->num_edges[u - 1] == 0) { u--; }
            u--;
        }
        else {
            // g->num_edges[u] == i
            while (g->num_edges[u + 1] - g->num_edges[u] == 0) {
                u++;
            }
        }
    }
    return u;
}

void free_graph(graph_t* g) {
    if (g->adj != nullptr)
        free(g->adj);

    if (g->num_edges != nullptr)
        free(g->num_edges);

    if (g->eid != nullptr)
        free(g->eid);
}


//Populate eid and edge list
void getEidAndEdgeList(graph_t* g, int* Edge_u, int* Edge_v) {
    //Allocate space for eid -- size g->m
    g->eid = (eid_t*)malloc(g->m * sizeof(eid_t));

    //Edge upper_tri_start of each edge
    auto* num_edges_copy = (eid_t*)malloc((g->n + 1) * sizeof(eid_t));
    assert(num_edges_copy != nullptr);

    //parallel construct

    auto* highdeg_node = (eid_t*)malloc((g->m) * sizeof(eid_t));
    assert(num_edges_copy != nullptr);

    num_edges_copy[0] = 0;
#pragma omp parallel
    {
#pragma omp for
        // Histogram (Count).
        for (vid_t u = 0; u < g->n; u++) {
            int deg_u = g->num_edges[u + 1] - g->num_edges[u];
            num_edges_copy[u + 1] = 0;
            for (eid_t k = g->num_edges[u]; k < g->num_edges[u + 1]; k++) {
                int deg_v = g->num_edges[g->adj[k] + 1] - g->num_edges[g->adj[k]];
                if (deg_v > deg_u) {
                    num_edges_copy[u + 1]++;
                    highdeg_node[k] = num_edges_copy[u + 1];
                }
                else if (deg_v == deg_u && g->adj[k] > u) {
                    num_edges_copy[u + 1]++;
                    highdeg_node[k] = num_edges_copy[u + 1];
                }
                else highdeg_node[k] = 0;
            }
        }
        // Scan.
#pragma omp single
        {
            for (auto i = 0; i < g->n; i++) {
                num_edges_copy[i + 1] += num_edges_copy[i];
            }
        }

        // Transform.
        auto u = 0;
#pragma omp for schedule(dynamic, 6000)
        for (eid_t j = 0; j < g->m; j++) {
            u = FindSrc(g, u, j);
            if (highdeg_node[j] == 0) {
                auto v = g->adj[j];
                auto offset = BranchFreeBinarySearch(g->adj, g->num_edges[v], g->num_edges[v + 1], u);
                auto eid = num_edges_copy[v] + highdeg_node[offset] - 1;
                g->eid[j] = eid;
                //idToEdge[eid] = Edge(v, u);
                Edge_u[eid] = v;
                Edge_v[eid] = u;
            }
            else {
                g->eid[j] = num_edges_copy[u] + highdeg_node[j] - 1;
            }
        }

#ifdef _DEBUG_
        // verify eid.
#pragma omp for
        for (eid_t j = 0; j < g->m / 2; j++) {
            int u = Edge_u[j];
            int v = Edge_v[j];
            //Edge edge = idToEdge[j];
            if (g->num_edges[u + 1] - g->num_edges[u] > g->num_edges[v + 1] - g->num_edges[v]) {
                fprintf(stderr, "Edge List Order Not Correct...\n");
                exit(-1);
            }
            //auto u = edge.u;
            //auto v = edge.v;
            if (BranchFreeBinarySearch(g->adj, g->num_edges[v], g->num_edges[v + 1], u) == g->num_edges[v + 1]) {
                fprintf(stderr, "Not Found u: %d in v: %d\n", u, v);
                exit(-1);
            }
            if (BranchFreeBinarySearch(g->adj, g->num_edges[u], g->num_edges[u + 1], v) == g->num_edges[u + 1]) {
                fprintf(stderr, "Not Found v: %d in u: %d\n", v, u);
                exit(-1);
            }
        }
#pragma omp single
        fprintf(stderr, "Pass EID checking...\n");
#endif
    }
    free(highdeg_node);
    free(num_edges_copy);
}





Graph::Graph(char* dir_cstr) {
    dir = string(dir_cstr);

    ReadDegree();
    ReadAdjacencyList();
    CheckInputGraph();
}

using namespace std::chrono;

void Graph::ReadDegree() {
    auto start = high_resolution_clock::now();

    ifstream deg_file(dir + string("/b_degree.bin"), ios::binary);
    int int_size;
    deg_file.read(reinterpret_cast<char*>(&int_size), 4);

    deg_file.read(reinterpret_cast<char*>(&nodemax), 4);
    deg_file.read(reinterpret_cast<char*>(&edgemax), 4);
    //    log_info("int size: %d, n: %s, m: %s", int_size, FormatWithCommas(nodemax).c_str(),
    //        FormatWithCommas(edgemax).c_str());

    degree.resize(static_cast<unsigned long>(nodemax));
    deg_file.read(reinterpret_cast<char*>(&degree.front()), sizeof(int) * nodemax);

    auto end = high_resolution_clock::now();
    fprintf(stderr, "read degree file time: %.3lf s\n", duration_cast<milliseconds>(end - start).count() / 1000.0);
}

void Graph::ReadAdjacencyList() {
    auto start = high_resolution_clock::now();
    ifstream adj_file(dir + string("/b_adj.bin"), ios::binary);

    // csr representation
    node_off = (uint32_t*)malloc(sizeof(uint32_t) * (nodemax + 1));
    edge_dst = static_cast<int*>(malloc(sizeof(int) * static_cast<uint64_t>(edgemax + 16)));

    string dst_v_file_name = dir + string("/b_adj.bin");
    auto dst_v_fd = open(dst_v_file_name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
    int* buffer = (int*)mmap(0, static_cast<uint64_t>(edgemax) * 4u, PROT_READ, MAP_PRIVATE, dst_v_fd, 0);

    // prefix sum
    node_off[0] = 0;
    for (auto i = 0u; i < nodemax; i++) { node_off[i + 1] = node_off[i] + degree[i]; }

    auto end = high_resolution_clock::now();
    fprintf(stderr,"malloc, and sequential-scan time: %.3lf s\n", duration_cast<milliseconds>(end - start).count() / 1000.0);
    // load dst vertices into the array
#pragma omp parallel for schedule(dynamic, 1000)
    for (auto i = 0u; i < nodemax; i++) {
        // copy to the high memory bandwidth mem
        for (uint64_t offset = node_off[i]; offset < node_off[i + 1]; offset++) {
            edge_dst[offset] = buffer[offset];
        }
        // inclusive
        degree[i]++;
    }
    munmap(buffer, static_cast<uint64_t>(edgemax) * 4u);

#ifdef _DEBUG_
    // Verify.
#pragma omp parallel for schedule(dynamic, 1000)
    for (auto u = 0u; u < nodemax; u++) {
        for (size_t offset = node_off[u]; offset < node_off[u + 1]; offset++) {
            auto v = edge_dst[offset];
            if (BranchFreeBinarySearch(edge_dst, node_off[v], node_off[v + 1], (int)u) == node_off[v + 1]) {
                fprintf(stderr,"CSR not correct\n");
                exit(-1);
            }
        }
    }
    fprintf(stderr,"CSR verify pass\n");
#endif

    auto end2 = high_resolution_clock::now();
    fprintf(stderr, "read adjacency list file time: %.3lf s\n", duration_cast<milliseconds>(end2 - end).count() / 1000.0);
}

/*
void Graph::ReadAdjacencyList() {
    auto begin = high_resolution_clock::now();
    FILE* adj_file = fopen((dir + string("/b_adj.bin")).c_str(), "rb");

    // csr representation
    node_off = (uint32_t*)malloc(sizeof(uint32_t) * (nodemax + 1));
    edge_dst = static_cast<int*>(malloc(sizeof(int) * static_cast<uint64_t>(edgemax + 16)));

    int* buffer = (int*)malloc(sizeof(int) * nodemax);

    // prefix sum
    node_off[0] = 0;
    // load dst vertices into the array
    for (auto i = 0u; i < nodemax; i++) {
        // copy to the high memory bandwidth mem
        if (degree[i] > 0) fread(buffer, sizeof(int), degree[i], adj_file);
        for (auto j = 0; j < degree[i]; j++) edge_dst[node_off[i] + j] = buffer[j];
        node_off[i + 1] = node_off[i] + degree[i];
        // inclusive
        degree[i]++;
    }
    free(buffer);
    fclose(adj_file);
#ifdef _DEBUG_
    // Verify.
#pragma omp parallel for schedule(dynamic, 1000)
    for (auto u = 0u; u < nodemax; u++) {
        for (size_t offset = node_off[u]; offset < node_off[u + 1]; offset++) {
            auto v = edge_dst[offset];
            if (BranchFreeBinarySearch(edge_dst, node_off[v], node_off[v + 1], (int)u) == node_off[v + 1]) {
                fprintf(stderr, "CSR not correct\n");
                exit(-1);
            }
        }
    }
    fprintf(stderr, "CSR verify pass\n");
#endif

    auto end2 = high_resolution_clock::now();
    fprintf(stderr, "read adjacency list file time: %.3lf s\n", duration_cast<milliseconds>(end2 - begin).count() / 1000.0);

}
*/

void Graph::CheckInputGraph() {
    auto start = high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 5000)
    for (auto i = 0u; i < nodemax; i++) {
        for (auto j = node_off[i]; j < node_off[i + 1]; j++) {
            if (edge_dst[j] == static_cast<int>(i)) {
                fprintf(stderr, "Self loop\n");
                exit(1);
            }
            if (j > node_off[i] && edge_dst[j] <= edge_dst[j - 1]) {
                fprintf(stderr, "Edges not sorted in increasing id order!\nThe program may not run properly!\n");
                exit(1);
            }
        }
    }
    auto end = high_resolution_clock::now();
    fprintf(stderr, "check input graph file time: %.3lf s\n", duration_cast<milliseconds>(end - start).count() / 1000.0);
}
