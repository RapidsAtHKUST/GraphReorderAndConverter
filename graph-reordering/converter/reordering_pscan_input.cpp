//
// Created by yche on 8/27/18.
//

#include <string>
#include <utils/yche_serialization.h>
#include <utils/log.h>

#include <cassert>
#include <algorithm>
#include <chrono>
#include <memory>

#include "pscan_graph.h"
#include "output_util.h"

using namespace std;
using namespace std::chrono;

void verify(ppscan::Graph &original_g, vector<int> &new_vid_dict, string &reorder_path) {
    ppscan::Graph g = ppscan::Graph(reorder_path.c_str());
    assert(g.nodemax == original_g.nodemax);
    assert(g.edgemax == original_g.edgemax);
    assert(g.degree.size() == original_g.degree.size());

    auto start = high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1000)
    for (auto u = 0; u < original_g.nodemax; u++) {
        auto new_u_id = new_vid_dict[u];

        assert(original_g.node_off[u + 1] - original_g.node_off[u] == g.node_off[new_u_id + 1] - g.node_off[new_u_id]);
        assert(original_g.degree[u] == g.degree[new_u_id]);
        for (auto offset = original_g.node_off[u]; offset < original_g.node_off[u + 1]; offset++) {
            auto v = original_g.edge_dst[offset];
            auto new_v_id = new_vid_dict[v];
            assert(binary_search(g.edge_dst + g.node_off[new_u_id], g.edge_dst + g.node_off[new_u_id + 1], new_v_id));
        }
    }
    auto end = high_resolution_clock::now();
    log_info("parallel verification time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
    log_info("correct");
}


int main(int argc, char *argv[]) {
    //set log file descriptor
#ifdef USE_LOG
    FILE *log_f;
    log_f = fopen(argv[3], "a+");
    log_set_fp(log_f);
#endif

    string dir = argv[1];
    string reorder_method = argv[2];

    vector<int32_t> new_vid_dict;

    // 2nd: read csr
    ppscan::Graph g = ppscan::Graph(dir.c_str());
    // read the reordering dictionary

    // degree-descending-reordering is used for reducing hash-join workloads for all intersection count computation
    if (reorder_method == "rev_deg") {
        vector<int> old_vid_dict(g.degree.size());
        for (auto i = 0; i < old_vid_dict.size(); i++) { old_vid_dict[i] = i; }

        // sort by degrees, large -> small
        sort(begin(old_vid_dict), end(old_vid_dict), [&g](int l, int r) -> bool { return g.degree[l] > g.degree[r]; });

        // construct the reordering dictionary
        new_vid_dict = vector<int>(g.degree.size());
        for (auto i = 0; i < g.degree.size(); i++) {
            new_vid_dict[old_vid_dict[i]] = i;
        }
    } else {
        string reorder_file_path = dir + "/" + reorder_method + ".dict";
        FILE *pFile = fopen(reorder_file_path.c_str(), "r");
        YcheSerializer serializer;
        serializer.read_array(pFile, new_vid_dict);
        fclose(pFile);
    }

    // 1st: read the reordering node dictionary, assert it is a permutation first
    for (auto i = 0; i < std::min<int32_t>(10, new_vid_dict.size()); i++) {
        log_info("%d", new_vid_dict[i]);
    }
    vector<int> verify_map(new_vid_dict.size(), 0);

    int cnt = 0;
#pragma omp parallel for reduction(+:cnt)
    for (auto i = 0; i < new_vid_dict.size(); i++) {
        if (verify_map[new_vid_dict[i]] == 0) {
            cnt++;
            verify_map[new_vid_dict[i]] = 1;
        } else {
            assert(false);
        }
    }
    log_info("%d, %d", cnt, new_vid_dict.size());
    assert(cnt == new_vid_dict.size());


    // reordering
    // old_vid_dict: new_id -> old_id, construct the inverted index (no duplicate)
    vector<int32_t> old_vid_dict(new_vid_dict.size());

    auto start = high_resolution_clock::now();

#pragma omp parallel for
    for (auto old_id = 0; old_id < new_vid_dict.size(); old_id++) {
        old_vid_dict[new_vid_dict[old_id]] = old_id;
    }

    vector<int> new_deg(g.nodemax);
    for (auto new_id = 0; new_id < g.nodemax; new_id++) { new_deg[new_id] = g.degree[old_vid_dict[new_id]]; }

    // CSR: new_off, new_neighbors
    vector<uint32_t> new_off(g.nodemax + 1);
    new_off[0] = 0;
    assert(new_off.size() == g.nodemax + 1);
    for (auto i = 0u; i < new_off.size(); i++) { new_off[i + 1] = new_off[i] + new_deg[i]; }

    vector<int> new_neighbors(g.edgemax);
    auto end = high_resolution_clock::now();
    log_info("init ordering structures time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

#pragma omp parallel for
    for (auto i = 0; i < g.nodemax; i++) {
        auto origin_i = old_vid_dict[i];
        // transform
        auto cur_idx = new_off[i];
        for (auto my_old_off = g.node_off[origin_i]; my_old_off < g.node_off[origin_i + 1]; my_old_off++) {
            assert(cur_idx <= g.edgemax);
            assert(my_old_off <= g.edgemax);
            assert(g.edge_dst[my_old_off] < g.nodemax);
            new_neighbors[cur_idx] = new_vid_dict[g.edge_dst[my_old_off]];
            cur_idx++;
        }
        // sort the local ranges
        sort(begin(new_neighbors) + new_off[i], begin(new_neighbors) + new_off[i + 1]);
    }
    auto end2 = high_resolution_clock::now();
    log_info("parallel transform and sort: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);

    // 3rd: output to the disk
    string cmd = string("mkdir -p ") + dir + "/" + string(reorder_method);
    log_info("cmd: %s", cmd.c_str());
    string info = exec(cmd.c_str());
    log_info("ret: %s", info.c_str());

    string reorder_deg_file_path = dir + "/" + reorder_method + "/" + "b_degree.bin";
    string reorder_adj_file_path = dir + "/" + reorder_method + "/" + "b_adj.bin";
    WriteToOutputFiles(reorder_deg_file_path, reorder_adj_file_path, new_deg, new_neighbors);

    // 4th: verify the correctness of reordering, check all the edges with the dictionary
    string reorder_path = string(dir + "/" + reorder_method);
    verify(g, new_vid_dict, reorder_path);
}
