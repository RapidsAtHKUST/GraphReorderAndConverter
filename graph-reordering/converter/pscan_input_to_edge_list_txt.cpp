//
// Created by yche on 8/26/18.
//

#include <chrono>
#include <cassert>

#include "../utils/log.h"
#include "../utils/yche_serialization.h"

#include "pscan_graph.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {
    //set log file descriptor
#ifdef USE_LOG
    FILE *log_f;
    if (argc >= 3) {
        log_f = fopen(argv[2], "a+");
        log_set_fp(log_f);
    }
#endif

    ppscan::Graph g(argv[1]);

    auto start = high_resolution_clock::now();

    // 1st: to edge list binary here
#ifdef DIRECTED
    vector<pair<int32_t, int32_t >> edges(g.edgemax);

#pragma omp parallel for schedule(dynamic, 5000)
    for (auto i = 0u; i < g.nodemax; i++) {
        for (auto n = g.node_off[i]; n < g.node_off[i + 1]; n++) {
            edges[n].first = i;
            edges[n].second = g.edge_dst[n];
        }
    }
#else
    vector<pair<int32_t, int32_t >> edges(g.edgemax / 2);
    vector<int32_t> partial_deg(g.nodemax);
    vector<int32_t> prefix_sum(g.nodemax);

#pragma omp parallel for
    for (auto i = 0u; i < g.nodemax; i++) {
        auto it_end = lower_bound(g.edge_dst + g.node_off[i], g.edge_dst + g.node_off[i + 1], i);
        partial_deg[i] = static_cast<int32_t>(it_end - (g.edge_dst + g.node_off[i]));
    }
    // exclusive-scan
    prefix_sum[0] = 0;
    for (int i = 0; i < g.nodemax - 1; i++) {
        prefix_sum[i + 1] = prefix_sum[i] + partial_deg[i];
    }
    assert(prefix_sum[g.nodemax - 1] + partial_deg[g.nodemax - 1] == g.edgemax / 2);

#pragma omp parallel for schedule(dynamic, 5000)
    for (auto u = 0u; u < g.nodemax; u++) {
        // only for u > N[u]
        for (auto offset = 0; offset < partial_deg[u]; offset++) {
            int write_loc = offset + prefix_sum[u];
            edges[write_loc].first = u;
            edges[write_loc].second = g.edge_dst[g.node_off[u] + offset];
        }
    }
#endif
    auto end = high_resolution_clock::now();
    log_info("construct time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

    // 2nd: output
    string my_path = string(argv[1]) + "/" + "edge_lst.txt";
    ofstream ofs(my_path);
    for (auto edge: edges) {
        ofs << edge.first << " " << edge.second << "\n";
    }

    auto end2 = high_resolution_clock::now();
    log_info("output time: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);

#ifdef USE_LOG
    if (argc >= 3) {
        fflush(log_f);
        fclose(log_f);
    }
#endif
}