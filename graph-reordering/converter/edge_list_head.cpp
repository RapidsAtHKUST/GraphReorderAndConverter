//
// Created by yche on 8/26/18.
//

#include <cassert>

#include <string>
#include <algorithm>

#include "../utils/yche_serialization.h"
#include "../utils/log.h"

#include "pscan_graph.h"

using namespace std;

int main(int argc, char *argv[]) {
    //set log file descriptor
#ifdef USE_LOG
    FILE *log_f;
    if (argc >= 3) {
        log_f = fopen(argv[2], "a+");
        log_set_fp(log_f);
    }
#endif


    // 1st: load the edge list
    string path = string(argv[1]) + "/undir_edge_list.bin";

    FILE *pFile = fopen(path.c_str(), "r");
    YcheSerializer serializer;
    vector<pair<int32_t, int32_t >> edges;
    serializer.read_array(pFile, edges);

    // peek 10 elements
    for (int i = 0; i < std::min<size_t>(10, edges.size()); i++) {
        log_info("%d, %d", edges[i].first, edges[i].second);
    }
    fclose(pFile);

    // 2nd: verify the correctness, 1) number complete, 2) no-err
    ppscan::Graph g(argv[1]);
    assert(g.edgemax == edges.size() * 2);

    volatile bool is_correct = true;
#pragma omp parallel for
    for (size_t i = 0; i < edges.size(); i++) {
        int u = edges[i].first;
        int v = edges[i].second;

#ifdef DIRECTED
        if (!binary_search(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], v)) {
#else
        if (!binary_search(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], v) ||
            !binary_search(g.edge_dst + g.node_off[v], g.edge_dst + g.node_off[v + 1], u)) {
#endif
            is_correct = false;
            log_info("%d, %d", u, v);
        }
    }
    assert(is_correct);
    log_info("Correct");

#ifdef USE_LOG
    fflush(log_f);
    fclose(log_f);
#endif
}