//
// Created by yche on 9/2/18.
//
#include "output_util.h"

#include <memory>
#include <chrono>
#include <vector>
#include <fstream>

#include "../utils/log.h"

using namespace std;
using namespace std::chrono;

std::string exec(const char *cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}


void WriteToOutputFiles(string &deg_output_file, string &adj_output_file, vector<int> &degrees,
                        vector<int32_t> &dst_vertices) {
    ofstream deg_ofs(deg_output_file, ios::binary);

    auto start = high_resolution_clock::now();

    int int_size = sizeof(int);
    uint32_t vertex_num = degrees.size();
    uint32_t edge_num = dst_vertices.size();
    deg_ofs.write(reinterpret_cast<const char *>(&int_size), 4);
    deg_ofs.write(reinterpret_cast<const char *>(&vertex_num), 4);
    deg_ofs.write(reinterpret_cast<const char *>(&edge_num), 4);
    deg_ofs.write(reinterpret_cast<const char *>(&degrees.front()), degrees.size() * 4u);
    auto end = high_resolution_clock::now();
    log_info("degree file write time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

    ofstream adj_ofs(adj_output_file, ios::binary);
    adj_ofs.write(reinterpret_cast<const char *>(&dst_vertices.front()), dst_vertices.size() * 4u);
    auto end2 = high_resolution_clock::now();
    log_info("adj file write time: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);
}