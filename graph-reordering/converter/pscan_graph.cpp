#include"pscan_graph.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstring>

#ifdef MEM_KIND
#include <hbwmalloc.h>
#endif

#include <chrono>
#include <algorithm>
#include <fcntl.h>

#ifdef TBB

#include <tbb/parallel_sort.h>

#endif

#include "../utils/log.h"
#include "../utils/yche_util.h"

using namespace chrono;

using namespace ppscan;

Graph::Graph(const char *dir_cstr) {
    dir = string(dir_cstr);
    log_debug("%s", dir.c_str());
    ReadDegree();
    ReadAdjacencyList();
    CheckInputGraph();
}

void Graph::ReadDegree() {
    auto start = high_resolution_clock::now();

    ifstream deg_file(dir + string("/b_degree.bin"), ios::binary);
    int int_size;
    deg_file.read(reinterpret_cast<char *>(&int_size), 4);

    deg_file.read(reinterpret_cast<char *>(&nodemax), 4);
    deg_file.read(reinterpret_cast<char *>(&edgemax), 4);
    log_info("int size: %d, n: %s, m: %s", int_size, FormatWithCommas(nodemax).c_str(),
             FormatWithCommas(edgemax).c_str());

    degree.resize(static_cast<unsigned long>(nodemax));
    deg_file.read(reinterpret_cast<char *>(&degree.front()), sizeof(int) * nodemax);

    auto end = high_resolution_clock::now();
    log_info("read degree file time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
}

void Graph::ReadAdjacencyList() {
    auto start = high_resolution_clock::now();
    ifstream adj_file(dir + string("/b_adj.bin"), ios::binary);

    // csr representation
    node_off = new uint32_t[nodemax + 1];
#if defined(MEM_KIND)
    //    edge_dst = new int[edgemax + 16];   // padding for simd
        // allocation on high-bandwidth memory (16GB)
        edge_dst = static_cast<int *>(hbw_malloc(sizeof(int) * static_cast<uint64_t>(edgemax + 16)));
#else
    edge_dst = static_cast<int *>(malloc(sizeof(int) * static_cast<uint64_t>(edgemax + 16)));
#endif

    string dst_v_file_name = dir + string("/b_adj.bin");
    auto dst_v_fd = open(dst_v_file_name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
    int *buffer = (int *) mmap(0, static_cast<uint64_t >(edgemax) * 4u, PROT_READ, MAP_PRIVATE, dst_v_fd, 0);

    // prefix sum
    node_off[0] = 0;
    for (auto i = 0; i < nodemax; i++) { node_off[i + 1] = node_off[i] + degree[i]; }

    auto end = high_resolution_clock::now();
    log_info("malloc, and sequential-scan time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
    // load dst vertices into the array
#pragma omp parallel for schedule(dynamic, 1000)
    for (auto i = 0; i < nodemax; i++) {
        // copy to the high memory bandwidth mem
        for (uint64_t offset = node_off[i]; offset < node_off[i + 1]; offset++) {
            edge_dst[offset] = buffer[offset];
        }
#ifdef LEGACY
        if (degree[i] > 0) {
            adj_file.read(reinterpret_cast<char *>(&edge_dst[node_off[i]]), degree[i] * sizeof(int));
        }
#endif
    }
    munmap(buffer, static_cast<uint64_t >(edgemax) * 4u);

    auto end2 = high_resolution_clock::now();
    log_info("read adjacency list file time: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);
}

void Graph::CheckInputGraph() {
    auto start = high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 10000)
    for (auto i = 0; i < nodemax; i++) {
        for (auto j = node_off[i]; j < node_off[i + 1]; j++) {
            if (edge_dst[j] == i) {
                cout << "Self loop\n";
                exit(1);
            }
            if (j > node_off[i] && edge_dst[j] <= edge_dst[j - 1]) {
                cout << "Edges not sorted in increasing id order!\nThe program may not run properly!\n";
                exit(1);
            }
        }
    }
    auto end = high_resolution_clock::now();
    log_info("check input graph file time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
}

uint32_t Graph::BinarySearch(int *array, uint32_t offset_beg, uint32_t offset_end, int val) {
    auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_beg) + offset_end) / 2);
    if (array[mid] == val) { return mid; }
    return val < array[mid] ? BinarySearch(array, offset_beg, mid, val) : BinarySearch(array, mid + 1, offset_end, val);
}
