//
// Created by yche on 9/2/18.
//

#include <chrono>
#include <cassert>
#include <algorithm>

#ifdef TBB

#include <tbb/parallel_sort.h>

#endif

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

int FindSrc(ppscan::Graph *g, int u, uint32_t edge_idx) {
// update last_u, preferring galloping instead of binary search because not large range here
    u = GallopingSearch(g->node_off, static_cast<uint32_t>(u) + 1, g->nodemax + 1, edge_idx);
// 1) first > , 2) has neighbor
    if (g->node_off[u] > edge_idx) {
        while (g->degree[u - 1] == 1) { u--; }
        u--;
    } else {
// g->node_off[u] == i
        while (g->degree[u] == 1) {
            u++;
        }
    }
    return u;
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

    vector<int> old_vid_dict;
    vector<int> new_deg;

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
            old_vid_dict = vector<int>(deg_lst.size());
        }

        // this step is for initialization for reordering
#pragma omp for
        for (auto i = 0; i < old_vid_dict.size(); i++) { old_vid_dict[i] = i; }
    }

    // extra 1) next steps for degree-descending... sort by degrees, large -> small
#ifdef TBB
    tbb::parallel_sort(begin(old_vid_dict), end(old_vid_dict),
                       [&deg_lst](int l, int r) -> bool { return deg_lst[l] > deg_lst[r]; });
#else
    sort(begin(old_vid_dict), end(old_vid_dict), [&deg_lst](int l, int r) -> bool { return deg_lst[l] > deg_lst[r]; });
#endif

    // extra 2) construct the reordering dictionary, update the degree
    vector<int> new_vid_dict = vector<int>(deg_lst.size());
#pragma omp parallel
    {
#pragma omp for
        for (auto i = 0; i < deg_lst.size(); i++) {
            new_vid_dict[old_vid_dict[i]] = i;
        }
#pragma omp single
        {
            new_deg = vector<int>(deg_lst.size());
        }
#pragma omp for
        for (auto new_id = 0; new_id < deg_lst.size(); new_id++) { new_deg[new_id] = deg_lst[old_vid_dict[new_id]]; }
    }

#ifdef DEBUG
    // read the reordering node dictionary, assert it is a permutation first
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
#endif

    // extra 3) prefix sum computation for CSR: new_off, new_neighbors
    vector<uint32_t> new_off(deg_lst.size() + 1);
    new_off[0] = 0;
    for (auto i = 0u; i < new_off.size(); i++) { new_off[i + 1] = new_off[i] + new_deg[i]; }

    // extra 4) mapping to the domain and sort local ranges
    vector<int> new_neighbors(adj_lst.size());
#pragma omp parallel for schedule(dynamic)
    for (auto new_vid = 0; new_vid < deg_lst.size(); new_vid++) {
        auto origin_i = old_vid_dict[new_vid];
        // transform
        auto cur_idx = new_off[new_vid];
        for (auto my_old_off = off[origin_i]; my_old_off < off[origin_i + 1]; my_old_off++) {
            new_neighbors[cur_idx] = new_vid_dict[adj_lst[my_old_off]];
            cur_idx++;
        }
        // sort the local ranges
        sort(begin(new_neighbors) + new_off[new_vid], begin(new_neighbors) + new_off[new_vid + 1]);
    }
    auto end2 = high_resolution_clock::now();
    log_info("edge list to csr time: %.3lf s", duration_cast<milliseconds>(end2 - start).count() / 1000.0);

    // =========== lastly: verify with reading another  =================================================
#ifndef TBB
    ino64_t err_cnt = 0;
    ppscan::Graph verify_g(string(dir + "/rev_deg").c_str());
#pragma omp parallel
    {
#pragma omp for
        for (auto i = 0; i < verify_g.nodemax; i++) {
            assert(new_deg[i] == verify_g.degree[i]);
        }
#pragma omp for
        for (auto i = 0; i < verify_g.nodemax; i++) {
            assert(new_off[i] == verify_g.node_off[i]);
        }
#pragma omp single
        log_info("new_off, new_deg correct");
#pragma omp for schedule(dynamic, 6000) reduction(+:err_cnt)
        for (auto i = 0; i < verify_g.edgemax; i++) {
            if (verify_g.edge_dst[i] != new_neighbors[i]) {
#pragma omp critical
                log_info("edge incorrect: %s, %d, %d", FormatWithCommas(i).c_str(), verify_g.edge_dst[i],
                         new_neighbors[i]);
                err_cnt++;
            }
            assert(verify_g.edge_dst[i] == new_neighbors[i]);
        }
    }
    if (err_cnt == 0)
        log_info("Correct");
    else
        log_info("err: %lld", err_cnt);
#endif
}