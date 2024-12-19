#include "Graph.h"

#include <cassert>


#include <sys/stat.h>
#include <fcntl.h>

#include <chrono>
#include <fstream>
#include <omp.h>

#include "search_util.h"
#include "parasort_cmp.h"



inline vid_t FindSrc(graph_t* g, vid_t u, eid_t edge_idx) {
    if (edge_idx >= g->num_edges[u + 1]) {
        // update last_u, preferring galloping instead of binary search because not large range here
        u = GallopingSearch(g->num_edges, u + 1, g->n + 1, edge_idx);
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
void getEidAndEdgeList(graph_t* g, vid_t* Edge_u, vid_t* Edge_v, eid_t* num_edges_copy) {
    //Allocate space for eid -- size g->m
    g->eid = (eid_t*)malloc(g->m * sizeof(eid_t));
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
    free(g->eid);
    g->eid=nullptr;
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
    deg_file.read(reinterpret_cast<char*>(&degree.front()), sizeof(vid_t) * nodemax);

    auto end = high_resolution_clock::now();
    fprintf(stderr, "read degree file time: %.3lf s\n", duration_cast<milliseconds>(end - start).count() / 1000.0);
}

void Graph::ReadAdjacencyList() {
    auto begin = high_resolution_clock::now();
    FILE* adj_file = fopen((dir + string("/b_adj.bin")).c_str(), "rb");

    // csr representation
    node_off = (eid_t*)malloc(sizeof(eid_t) * (nodemax + 1));
    edge_dst = static_cast<vid_t*>(malloc(sizeof(vid_t) * edgemax));// static_cast<uint64_t>(edgemax + 16)));

    vid_t* buffer = (vid_t*)malloc(sizeof(vid_t) * nodemax);//sizeof uint64_ change to int

    // prefix sum
    node_off[0] = 0;
    // load dst vertices into the array
    for (auto i = 0u; i < nodemax; i++) {
        // copy to the high memory bandwidth mem
        if (degree[i] > 0) fread(buffer, sizeof(vid_t), degree[i], adj_file);//sizeof uint64_ change to int
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
            if (BranchFreeBinarySearch(edge_dst, node_off[v], node_off[v + 1], (vid_t)u) == node_off[v + 1]) {
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

void Graph::CheckInputGraph() {
    auto start = high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 5000)
    for (auto i = 0u; i < nodemax; i++) {
        for (auto j = node_off[i]; j < node_off[i + 1]; j++) {
            if (edge_dst[j] == static_cast<vid_t>(i)) {
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



void Reorder(graph_t& g, vector<vid_t>& new_vid_dict, vector<vid_t>& old_vid_dict) {
    auto start = high_resolution_clock::now();

    new_vid_dict = vector<vid_t>(g.n);
    for (auto i = 0; i < g.n; i++) {
        new_vid_dict[old_vid_dict[i]] = i;
    }
    // new-deg
    vector<vid_t> new_deg(g.n);
    for (auto new_id = 0; new_id < g.n; new_id++) {
        auto vertex = old_vid_dict[new_id];
        new_deg[new_id] = g.num_edges[vertex + 1] - g.num_edges[vertex];
        assert(new_deg[new_id] >= 0);
    }

    // verify permutation
    for (auto i = 0; i < std::min<vid_t>(5, static_cast<vid_t>(new_vid_dict.size())); i++) {
        fprintf(stderr, "old->new %d -> %d", i, new_vid_dict[i]);
    }
    vector<vid_t> verify_map(new_vid_dict.size(), 0);
    vid_t cnt = 0;

#pragma omp parallel
    {
#pragma omp for reduction(+:cnt)
        for (auto i = 0; i < new_vid_dict.size(); i++) {
            if (verify_map[new_vid_dict[i]] == 0) {
                cnt++;
                verify_map[new_vid_dict[i]] = 1;
            }
            else {
                assert(false);
            }
        }
#pragma omp single
        fprintf(stderr, "%d, %d", cnt, new_vid_dict.size());
        assert(cnt == new_vid_dict.size());
    }
    // 1st CSR: new_off, new_neighbors
    vector<eid_t> new_off(g.n + 1);
    new_off[0] = 0;
    assert(new_off.size() == g.n + 1);
    for (auto i = 0u; i < g.n; i++) { new_off[i + 1] = new_off[i] + new_deg[i]; }
    fprintf(stderr, "%zu", new_off[g.n]);
    assert(new_off[g.n] == g.m);

    vector<vid_t> new_neighbors(g.m);

    //log_info("init ordering structures time: %.9lf s", timer.elapsed_and_reset());

    // 2nd Parallel Transform
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 100)
        for (auto i = 0; i < g.n; i++) {
            auto origin_i = old_vid_dict[i];
            // transform
            auto cur_idx = new_off[i];
            for (auto my_old_off = g.num_edges[origin_i]; my_old_off < g.num_edges[origin_i + 1]; my_old_off++) {
                if (cur_idx > g.m) {
                    fprintf(stderr, "%d, i: %d", cur_idx, i);
                }
                assert(cur_idx <= g.m);
                assert(my_old_off <= g.m);
                assert(g.adj[my_old_off] < g.n);
                new_neighbors[cur_idx] = new_vid_dict[g.adj[my_old_off]];
                cur_idx++;
            }
            // sort the local ranges
            sort(begin(new_neighbors) + new_off[i], begin(new_neighbors) + new_off[i + 1]);
        }
    }
    auto end = high_resolution_clock::now();
    fprintf(stderr, "parallel transform and sort: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

    memcpy(g.adj, &new_neighbors.front(), g.m * sizeof(vid_t));
    memcpy(g.num_edges, &new_off.front(), (g.n + 1) * sizeof(eid_t));
}

void ReorderDegAscending(graph_t& g, vector<vid_t>& new_vid_dict, vector<vid_t>& old_vid_dict) {
    auto start = high_resolution_clock::now();

    old_vid_dict = vector<vid_t>(g.n);
    for (auto i = 0; i < old_vid_dict.size(); i++) { old_vid_dict[i] = i; }

    parasort(old_vid_dict.size(), &old_vid_dict.front(),
        [&g](vid_t l, vid_t r) -> bool {
            return g.num_edges[l + 1] - g.num_edges[l] < g.num_edges[r + 1] - g.num_edges[r];
        },
        omp_get_max_threads());

    auto end = high_resolution_clock::now();
    fprintf(stderr, "Deg-Ascending time:  %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

    Reorder(g, new_vid_dict, old_vid_dict);
}

int GraphDivide(graph_t* g, uint64_t MemorySize, vid_t*& cut_id) {
    MemorySize = MemorySize - 4 * (g->n) * sizeof(vid_t);
    eid_t max_deg = 0;
#pragma omp parallel for
    for (auto i = 0; i < g->n; i++)
    {
        auto temp = g->num_edges[i + 1] - g->num_edges[i];
#pragma omp critical
        {
            if (temp > max_deg) max_deg = temp;
        }
    }
    if (MemorySize < 4 * max_deg * sizeof(vid_t)) {
        fprintf(stderr, "GPU Memory is not enough!!!");
        exit(1);
    }
    uint64_t GraphSize = (g->n + 1) * sizeof(eid_t) + (g->m) * (3 * sizeof(vid_t) + sizeof(char));
    int pn = 1;
    for (auto s = GraphSize; s > MemorySize; s = GraphSize / pn)
    {
        pn++;
    }
    cut_id = (vid_t*)malloc((pn + 1) * sizeof(vid_t));
    cut_id[0] = 0;
    cut_id[pn] = g->n;
    int p_count = 1;
    eid_t p_length = g->m / pn;
    for (auto i = 0; i < g->n; i++)
    {
        if (g->num_edges[i] > p_count * p_length - 1) {
            cut_id[p_count] = i;
            p_count++;
            if (p_count >= pn) break;
        }
    }
    return pn;
}

void LoadGraph(GraphChip* gc, graph_t* g, vid_t* cut_id, vid_t* Edge_u,
    vid_t* Edge_v, eid_t* num_edges_copy, int pn) {
    for (auto i = 0; i < pn; i++)
    {
        vid_t num_others = 0;
        vid_t num_nodes = cut_id[i + 1] - cut_id[i];
        vector<bool> flag(g->n, false);
#pragma omp parallel
        {
#pragma omp for
            for (auto j = num_edges_copy[cut_id[i]]; j < num_edges_copy[cut_id[i + 1]]; j++)
            {
                if (Edge_v[j] >= cut_id[i + 1]) flag[Edge_v[j]] = true;
            }
#pragma omp for reduction(+:num_others)
            for (auto j = 0; j < g->n; j++)
            {
                if (flag[j]) num_others++;
            }
        }
        vector<vid_t> nodes(num_nodes + num_others);// nodes in this subgraph
        gc[i].node_index = (vid_t*)malloc(g->n * sizeof(vid_t));
        memset(gc[i].node_index, -1, g->n * sizeof(vid_t));
        gc[i].num_edges = (eid_t*)malloc((num_nodes + num_others + 1) * sizeof(eid_t));
        gc[i].num_edges[0] = 0u;
        gc[i].estart = (eid_t*)malloc((num_nodes + 1) * sizeof(eid_t));
        gc[i].estart[0] = 0u;
        //for (auto j = num_edges_copy[cut_id[i]]; j < num_edges_copy[cut_id[i + 1]]; j++)
        //{
        //    if (Edge_v[j] >= cut_id[i + 1]) others.push_back(Edge_v[j]);
        //}
#pragma omp parallel for
        for (auto j = 0; j < num_nodes; j++)
        {
            nodes[j] = cut_id[i] + j;
            gc[i].num_edges[j + 1] = g->num_edges[cut_id[i] + j + 1] - g->num_edges[cut_id[i]];
            gc[i].node_index[cut_id[i] + j] = j;
            gc[i].estart[j + 1] = num_edges_copy[cut_id[i] + j + 1] - num_edges_copy[cut_id[i]];
        }
        gc[i].n = num_nodes + num_others;
        gc[i].m = num_edges_copy[cut_id[i + 1]] - num_edges_copy[cut_id[i]];
        gc[i].el_u = (vid_t*)malloc(gc[i].m * sizeof(vid_t));
        gc[i].el_v = (vid_t*)malloc(gc[i].m * sizeof(vid_t));
        gc[i].similar = (char*)malloc(gc[i].m * sizeof(char));
        //memset(gc[i].similar, 'U', gc[i].m * sizeof(char));
        if (num_others > 0) {
            //others array
            //vid_t* others = (vid_t*)malloc(num_others * sizeof(vid_t));
            vid_t pos = num_nodes;
            for (auto j = 0; j < g->n; j++)
            {
                if (flag[j]) {
                    nodes[pos] = j;
                    gc[i].node_index[j] = pos;
                    pos++;
                }
            }
            //others deg
            vector<vid_t> others_deg(num_others + 1);
            others_deg[0] = gc[i].num_edges[num_nodes];
#pragma omp parallel
            {
#pragma omp for
                for (auto k = 0; k < num_others; k++)
                {
                    others_deg[k + 1] = g->num_edges[nodes[num_nodes + k] + 1]
                        - g->num_edges[nodes[num_nodes + k]];
                }
#pragma omp single
                {
                    for (auto k = 0; k < num_others; k++) {
                        others_deg[k + 1] += others_deg[k];
                    }
                }
#pragma omp for
                for (auto k = 0; k < num_others; k++)
                {
                    gc[i].num_edges[num_nodes + k + 1] = others_deg[k + 1];
                }
            }
        }
        gc[i].adj_length = gc[i].num_edges[num_nodes + num_others];
        gc[i].adj = (vid_t*)malloc(gc[i].adj_length * sizeof(vid_t));
#pragma omp parallel
        {
#pragma omp for
            for (auto j = 0; j < nodes.size(); j++)
            {
                auto start = g->num_edges[nodes[j]];
                auto length = g->num_edges[nodes[j] + 1] - start;
                for (auto k = 0; k < length; k++)
                {
                    gc[i].adj[gc[i].num_edges[j] + k] = g->adj[start + k];
                }
            }
#pragma omp for
            for (auto j = num_edges_copy[cut_id[i]]; j < num_edges_copy[cut_id[i + 1]]; j++)
            {
                gc[i].el_u[j - num_edges_copy[cut_id[i]]] = Edge_u[j];
                gc[i].el_v[j - num_edges_copy[cut_id[i]]] = Edge_v[j];
            }
        }
    }
}

void free_GraphChip(GraphChip* gc, int pn) {
    if (gc != nullptr && pn > 0) {
        for (auto i = 0; i < pn; i++)
        {
            if (gc[i].adj != nullptr) free(gc[i].adj);
            if (gc[i].num_edges != nullptr) free(gc[i].num_edges);
            if (gc[i].estart != nullptr) free(gc[i].estart);
            if (gc[i].el_u != nullptr) free(gc[i].el_u);
            if (gc[i].el_v != nullptr) free(gc[i].el_v);
            if (gc[i].node_index != nullptr) free(gc[i].node_index);
            if (gc[i].similar != nullptr) free(gc[i].similar);
        }
    }
}
