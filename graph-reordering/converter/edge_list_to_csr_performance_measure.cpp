//
// Created by yche on 9/2/18.
//

#include <chrono>
#include <cassert>
#include <algorithm>

#include "../utils/log.h"

#include "../utils/yche_serialization.h"

#include "output_util.h"
#include "pscan_graph.h"
#include "../utils/yche_util.h"

#include "omp.h"

using namespace std;
using namespace std::chrono;
#define ATOMIC

#ifdef MEM_KIND
using edge_lst_type = vector<pair<int32_t, int32_t>, hbw::allocator<pair<int32_t, int32_t>>>;
#else
using edge_lst_type = vector<pair<int32_t, int32_t>>;
#endif

edge_lst_type ReadEdgeList(string &dir) {
    edge_lst_type edges;
    string graphpath = dir + "/" + "undir_edge_list.bin";
    FILE *pFile = fopen(graphpath.c_str(), "r");
    YcheSerializer serializer;
    serializer.read_array(pFile, edges);
    fclose(pFile);
    return edges;
}

int main(int argc, char *argv[]) {
#ifdef USE_LOG
    FILE *log_f;
    if (argc >= 3) {
        log_f = fopen(argv[2], "a+");
        log_set_fp(log_f);
    }
#endif
    string dir(argv[1]);

    // 1st: read edge list, assume already no self-loop, duplicate edge
    auto load_start = high_resolution_clock::now();

    auto edge_lst = ReadEdgeList(dir);
    int32_t max_node_id = -1;
    vector<int32_t> deg_lst;
    vector<uint32_t> off;
    vector<int32_t> adj_lst;
    vector<uint32_t> cur_write_off;
    auto load_end = high_resolution_clock::now();
    log_info("load edge list bin time: %.3lf s",
             duration_cast<milliseconds>(load_end - load_start).count() / 1000.0);

    auto start = high_resolution_clock::now();
#if defined(LOCKS)
    omp_lock_t *locks;
#endif

#pragma omp parallel
    {
        // 1st: get the cardinality of degree array
#pragma omp for reduction(max: max_node_id)
        for (uint32_t i = 0; i < edge_lst.size(); i++) {
            max_node_id = max(max_node_id, max(edge_lst[i].first, edge_lst[i].second));
        }
#pragma omp single nowait
        {
            deg_lst = vector<int32_t>(static_cast<uint32_t>(max_node_id + 1));

        }
#pragma omp single nowait
        {
            off = vector<uint32_t>(static_cast<uint32_t>(max_node_id + 2));
            off[0] = 0;
        }
#pragma omp for
        for (auto i = 0; i < deg_lst.size(); i++) {
            deg_lst[i] = 0;
        }

        // 2nd: to count grouped neighbors, store in `deg_lst`
#pragma omp for
        for (uint32_t i = 0; i < edge_lst.size(); i++) {
            // atomic add for edge.first
            auto src = edge_lst[i].first;
            auto dst = edge_lst[i].second;
            int inc_deg_val, cur_deg_src;
            do {
                cur_deg_src = deg_lst[src];
                inc_deg_val = cur_deg_src + 1;
            } while (!__sync_bool_compare_and_swap(&(deg_lst[src]), cur_deg_src, inc_deg_val));
            do {
                cur_deg_src = deg_lst[dst];
                inc_deg_val = cur_deg_src + 1;
            } while (!__sync_bool_compare_and_swap(&(deg_lst[dst]), cur_deg_src, inc_deg_val));
        }

        // 3rd: compute prefix_sum and then scatter
#pragma omp single
        {
            for (int i = 0; i < deg_lst.size(); i++) {
                off[i + 1] = off[i] + deg_lst[i];
            }
        }

#pragma omp single nowait
        {
            adj_lst = vector<int32_t>(off[off.size() - 1]);
        }

#pragma omp single nowait
        {
            cur_write_off = off;
        }

#pragma omp single nowait
        {
#if defined(LOCKS)
            locks = new omp_lock_t[deg_lst.size()];
#endif
        }

#if defined(LOCKS)
#pragma omp for
        for (int i = 0; i < deg_lst.size(); i++) {
            omp_init_lock(&locks[i]);
        }
#endif

        // 4th: barrier before we do the computation, and then construct destination vertices in CSR
#pragma omp single
        {
            auto middle = high_resolution_clock::now();
            log_info("before csr transform time: %.3lf s",
                     duration_cast<milliseconds>(middle - start).count() / 1000.0);
        }

#pragma omp for
        for (uint32_t i = 0; i < edge_lst.size(); i++) {
            auto src = edge_lst[i].first;
            auto dst = edge_lst[i].second;

            uint32_t new_offset, old_offset;
#if defined(ATOMIC)
            do {
                old_offset = cur_write_off[src];
                new_offset = old_offset + 1;
            } while (!__sync_bool_compare_and_swap(&(cur_write_off[src]), old_offset, new_offset));
            adj_lst[old_offset] = dst;

            do {
                old_offset = cur_write_off[dst];
                new_offset = old_offset + 1;
            } while (!__sync_bool_compare_and_swap(&(cur_write_off[dst]), old_offset, new_offset));
            adj_lst[old_offset] = src;
#elif defined(LOCKS)
            omp_set_lock(&locks[src]);
            old_offset = cur_write_off[src];
            new_offset = old_offset + 1;
            cur_write_off[src] = new_offset;
            omp_unset_lock(&locks[src]);

            adj_lst[old_offset] = dst;

            omp_set_lock(&locks[dst]);
            old_offset = cur_write_off[dst];
            new_offset = old_offset + 1;
            cur_write_off[dst] = new_offset;
            omp_unset_lock(&locks[dst]);

            adj_lst[old_offset] = src;
#endif
        }
#pragma omp single
        {
            auto middle2 = high_resolution_clock::now();
            log_info("before sort time: %.3lf s",
                     duration_cast<milliseconds>(middle2 - start).count() / 1000.0);
        }
        // 5th: sort each local ranges
#pragma omp for schedule(dynamic)
        for (auto u = 0; u < deg_lst.size(); u++) {
            assert(cur_write_off[u] == off[u + 1]);
            sort(begin(adj_lst) + off[u], begin(adj_lst) + off[u + 1]);
        }
    }
    auto end = high_resolution_clock::now();
    log_info("edge list to csr time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);


    // lastly: verify with reading another
    ppscan::Graph verify_g(dir.c_str());
#pragma omp parallel for
    for (auto i = 0; i < verify_g.nodemax; i++) {
        assert(deg_lst[i] == verify_g.degree[i]);
    }
#pragma omp parallel for schedule(dynamic, 6000)
    for (auto i = 0; i < verify_g.edgemax; i++) {
        if (verify_g.edge_dst[i] != adj_lst[i]) {
#pragma omp critical
            log_info("edge incorrect: %s, %d, %d", FormatWithCommas(i).c_str(), verify_g.edge_dst[i], adj_lst[i]);
        }
        assert(verify_g.edge_dst[i] == adj_lst[i]);
    }
    log_info("Correct");

#ifdef OUTPUT_TO_FILE
    // output for verification
    string reorder_deg_file_path = dir + "/" + "raw" + "/" + "b_degree.bin";
    string reorder_adj_file_path = dir + "/" + "raw" + "/" + "b_adj.bin";
    string cmd = string("mkdir -p ") + dir + "/" + "raw";
    log_info("cmd: %s", cmd.c_str());
    string info = exec(cmd.c_str());
    log_info("ret: %s", info.c_str());

    WriteToOutputFiles(reorder_deg_file_path, reorder_adj_file_path, deg_lst, adj_lst);
#endif
}