//
// Created by yche on 9/2/18.
//

#ifndef SRC_OUTPUT_UTIL_H
#define SRC_OUTPUT_UTIL_H

#include <string>
#include <vector>

using namespace std;

std::string exec(const char *cmd);

void WriteToOutputFiles(string &deg_output_file, string &adj_output_file, vector<int> &degrees,
                        vector<int32_t> &dst_vertices);

#endif //SRC_OUTPUT_UTIL_H
