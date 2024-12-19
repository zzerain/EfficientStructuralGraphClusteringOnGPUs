#pragma once

#include <unordered_map>
#include <vector>
#include <map>


using namespace std;

//typedef unsigned int vid_t;
typedef int vid_t;
typedef unsigned int eid_t;

typedef struct {
    vid_t n;
    eid_t m;

    vid_t* adj;
    eid_t* num_edges;
    eid_t* eid;
    vector<vid_t> degree;
} graph_t;

//Define an Edge data type
struct Edge {
    int u;
    int v;

    Edge() {
        this->u = 0;
        this->v = 0;
    }

    Edge(vid_t u, vid_t v) {
        this->u = u;
        this->v = v;
    }
};

void free_graph(graph_t* g);

void getEidAndEdgeList(graph_t* g, vid_t* Edge_u, vid_t* Edge_v, eid_t* num_edges_copy);

void ReorderDegAscending(graph_t& g, vector<vid_t>& new_vid_dict, vector<vid_t>& old_vid_dict);



struct Graph {
    string dir;

    vid_t nodemax;//?
    eid_t edgemax;

    // csr representation
    eid_t* node_off;
    vid_t* edge_dst;

    vector<vid_t> degree;

    explicit Graph(char* dir_cstr);

public:
    void ReadDegree();

    void CheckInputGraph();

    void ReadAdjacencyList();
};

typedef struct {
    vid_t n;
    eid_t begin_id;
    eid_t m;
    eid_t adj_length;

    vid_t* adj;
    eid_t* num_edges;
    //eid_t* estart;
    vector<bool> flag;
    vid_t* nodes;
    vid_t* node_index;
    vid_t* el_u;
    vid_t* el_v;
    char* similar;
} GraphChip;

//vid_t* GraphDivide(graph_t* g, int pn);
void GraphDivide(graph_t* g, vid_t* Edge_u, vid_t* Edge_v,
    uint64_t capacity, vector<GraphChip*>& gc, int& pn);

//void LoadGraph(GraphChip* gc, graph_t* g, vid_t* cut_id, vid_t* Edge_u, 
//    vid_t* Edge_v, eid_t* num_edges_copy, int pn);

void LoadGraphEdge(vector<GraphChip*> &gc, graph_t* g, vid_t*& Edge_u,
    vid_t*& Edge_v, int pn);

void loadSpeGraphChip(vector<GraphChip*> &gc, graph_t* g, int i);

void freeSpeGraphChip(vector<GraphChip*> &gc, int i);

void LoadGraphCSR(vector<GraphChip*> gc, graph_t* g, int begin, int pn);

void free_GraphChip(vector<GraphChip*>& gc, int pn);
