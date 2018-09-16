/**
Copyright (c) 2015, Intel Corporation. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Intel Corporation nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*!
 * \brief Example of cache-locality optimizing reorderings.
 *
 * \ref "Parallelization of Reordering Algorithms for Bandwidth and Wavefront
 *       Reduction", Karantasis et al., SC 2014
 * \ref "AN OBJECT-ORIENTED ALGORITHMIC LABORATORY FOR ORDERING SPARSEMATRICES",
 *       Kumfert
 * \ref "Fast and Efficient Graph Traversal Algorithms for CPUs: Maximizing
 *       Single-Node Efficiency", Chhugani et al., IPDPS 2012
 * \ref "Multi-core spanning forest algorithms using the disjoint-set data
 *       structure", Patwary et al., IPDPS 2012
 *
 * Expected performance
   (web-Google.mtx can be downloaded from U of Florida matrix collection)
   BW can change run to run.

 In a 18-core Xeon E5-2699 v3 @ 2.3GHz, 56 gbps STREAM BW

OMP_NUM_THREADS=18 KMP_AFFINITY=granularity=fine,compact,1 test/reordering_test web-Google.mtx
/home/jpark103/matrices/web-Google.mtx:::symmetric m=916428 n=916428 nnz=8644102
original bandwidth 915881
SpMV BW   17.07 gbps
BFS reordering
Constructing permutation takes 0.0301461 (1.15 gbps)
Permute takes 0.0295999 (3.50 gbps)
Permuted bandwidth 557632
SpMV BW   37.77 gbps
RCM reordering w/o source selection heuristic
Constructing permutation takes 0.0886319 (0.39 gbps)
Permute takes 0.0256741 (4.04 gbps)
Permuted bandwidth 321046
SpMV BW   43.52 gbps
RCM reordering
Constructing permutation takes 0.143199 (0.24 gbps)
Permute takes 0.0248771 (4.17 gbps)
Permuted bandwidth 330214
SpMV BW   41.32 gbps

 */

#include <omp.h>

#ifdef MKL
#include <mkl.h>
#endif

#include "Utils.hpp"

#include <algorithm>
#include <utils/log.h>
#include <utils/yche_serialization.h>

#include "reordering/other-reorderings/SpMP/CSR.hpp"

using namespace std;
using namespace SpMP;

static const size_t LLC_CAPACITY = 32 * 1024 * 1024;
static const double *bufToFlushLlc = NULL;

void flushLlc() {
    double sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < LLC_CAPACITY / sizeof(bufToFlushLlc[0]); ++i) {
        sum += bufToFlushLlc[i];
    }
    FILE *fp = fopen("/dev/null", "w");
    fprintf(fp, "%f", sum);
    fclose(fp);
}

typedef enum {
    BFS = 0,
    RCM_WO_SOURCE_SELECTION,
    RCM,
} Option;


void WriteToFile(string &my_path, int *p, int n) {
    log_info("%s", my_path.c_str());
    FILE *pFile = fopen(my_path.c_str(), "wb");
    YcheSerializer serializer;
    serializer.write_array(pFile, p, n);

    // flush and close the file handle
    fflush(pFile);
    fclose(pFile);
}

int main(int argc, char **argv) {
#ifdef USE_LOG
    FILE *log_f;
    if (argc >= 3) {
        log_f = fopen(argv[2], "a+");
        log_set_fp(log_f);
    }
#endif

    if (argc < 2) {
        log_error("Usage: reordering_test matrix_in_matrix_market_format");
        return -1;
    }


    auto *A = new CSR(argv[1], 0, true /* force-symmetric */);
    auto nnz = A->getNnz();
    double bytes = (double) (sizeof(double) + sizeof(int)) * nnz + sizeof(double) * (A->m + A->n);
    log_info("m = %d nnz = %lld %f bytes = %f", A->m, nnz, (double) nnz / A->m, bytes);

    log_info("original bandwidth %d", A->getBandwidth());

    auto *x = MALLOC(double, A->m);
    auto *y = MALLOC(double, A->m);

    // allocate a large buffer to flush out cache
    bufToFlushLlc = (double *) _mm_malloc(LLC_CAPACITY, 64);
    flushLlc();

    auto *perm = MALLOC(int, A->m);
    auto *inversePerm = MALLOC(int, A->m);

    for (int o = BFS; o <= RCM; ++o) {
        auto option = (Option) o;

        switch (option) {
            case BFS:
                log_info("BFS reordering");
                break;
            case RCM_WO_SOURCE_SELECTION:
                log_info("RCM reordering w/o source selection heuristic");
                break;
            case RCM:
                log_info("RCM reordering");
                break;
            default:
                assert(false);
                break;
        }

        double t = -omp_get_wtime();
        string my_path;
//        string my_inverse_path;
        switch (option) {
            case BFS:
                A->getBFSPermutation(perm, inversePerm);
                my_path = string(argv[1]) + "/" + "bfs.dict";
//                my_inverse_path = string(argv[1]) + "bfs_inverse.dict";
                break;
            case RCM_WO_SOURCE_SELECTION:
                A->getRCMPermutation(perm, inversePerm, false);
                my_path = string(argv[1]) + "/" + "rcm_wo_src_sel.dict";
//                my_inverse_path = string(argv[1]) + "/" + "rcm_wo_src_sel_inverse.dict";
                break;
            case RCM:
                A->getRCMPermutation(perm, inversePerm);
                my_path = string(argv[1]) + "/" + "rcm_with_src_sel.dict";
//                my_inverse_path = string(argv[1]) + "/" + "rcm_with_src_sel_inverse.dict";
                break;
        }
        t += omp_get_wtime();

        log_info("Constructing permutation takes %g (%.2f gbps)", t, nnz * 4 / t / 1e9);

        isPerm(perm, A->m);
        isPerm(inversePerm, A->m);
        log_info("Finish Get Permutation...");

        // output to files

        WriteToFile(my_path, perm, A->m);
//        WriteToFile(my_inverse_path, inversePerm, A->m);
    }

    FREE(x);
    FREE(y);

    delete A;
}
