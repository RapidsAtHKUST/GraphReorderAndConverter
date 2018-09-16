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

#include <cstdio>
#include <cstring>
#include <climits>
//#include <unistd.h>

#include <algorithm>

#include <omp.h>

#include "CSR.hpp"
#include "Utils.hpp"
#include "MemoryPool.hpp"
#include "../../../converter/pscan_graph.h"

using namespace std;

namespace SpMP {

    bool CSR::useMemoryPool_() const {
        return MemoryPool::getSingleton()->contains(rowptr);
    }

    void CSR::alloc(int m, int nnz) {
        this->m = m;

        rowptr = MALLOC(offset_type, m + 1);
        colidx = MALLOC(int, nnz);

        assert(rowptr != NULL);
        assert(colidx != NULL);

        ownData_ = true;
    }

    CSR::CSR(const CSR &A) : m(A.m), n(A.n), ownData_(true) {
        int nnz = A.getNnz();

        rowptr = MALLOC(offset_type, m + 1);
        colidx = MALLOC(int, nnz);


        copyVector(rowptr, A.rowptr, A.m + 1);
        copyVector(colidx, A.colidx, nnz);
    }

    CSR::CSR(const char *fileName, int base /*=0*/, bool forceSymmetric /*=false*/, int pad /*=1*/)
            : rowptr(NULL), colidx(NULL), ownData_(false) {
        ppscan::Graph ppscan_graph(fileName);
        rowptr = ppscan_graph.node_off;
        colidx = ppscan_graph.edge_dst;
        m = ppscan_graph.nodemax;
        n = ppscan_graph.nodemax;
//        usleep(8000000);
    }

    CSR::CSR(int m, int n, offset_type *rowptr, int *colidx) :
            m(m), n(n), rowptr(rowptr), colidx(colidx), ownData_(false) {
        assert(getBase() == 0 || getBase() == 1);
    }

    void CSR::dealloc() {
        if (useMemoryPool_()) {
            // a large single contiguous chunk is allocated to
            // buffers except rowptr and colidx.
            rowptr = NULL;
            colidx = NULL;
        } else {
            if (ownData_) {
                FREE(rowptr);
                FREE(colidx);
            }
        }
    }

    CSR::~CSR() {
        dealloc();
    }

    static const int MAT_FILE_CLASSID = 1211216;

    void CSR::loadBin(const char *file_name, int base /*=0*/) {
        dealloc();

        FILE *fp = fopen(file_name, "r");
        if (!fp) {
            fprintf(stderr, "Failed to open %s\n", file_name);
            return;
        }

        int id;
        fread(&id, sizeof(id), 1, fp);
        if (MAT_FILE_CLASSID != id) {
            fprintf(stderr, "Wrong file ID (%d)\n", id);
        }

        fread(&m, sizeof(m), 1, fp);
        fread(&n, sizeof(n), 1, fp);
        int nnz;
        fread(&nnz, sizeof(nnz), 1, fp);

        alloc(m, nnz);

        fread(rowptr + 1, sizeof(rowptr[0]), m, fp);
        rowptr[0] = 0;
        for (int i = 1; i < m; ++i) {
            rowptr[i + 1] += rowptr[i];
        }

        fread(colidx, sizeof(colidx[0]), nnz, fp);

        fclose(fp);

        if (1 == base) {
            make1BasedIndexing();
        } else {
            assert(0 == base);
        }
    }

    void CSR::make0BasedIndexing() {
        if (0 == getBase()) return;

        int nnz = getNnz();

#pragma omp parallel for
        for (int i = 0; i <= m; i++)
            rowptr[i]--;

#pragma omp parallel for
        for (int i = 0; i < nnz; i++)
            colidx[i]--;
    }

    void CSR::make1BasedIndexing() {
        if (1 == getBase()) return;

        int nnz = getNnz();

#pragma omp parallel for
        for (int i = 0; i <= m; i++)
            rowptr[i]++;

#pragma omp parallel for
        for (int i = 0; i < nnz; i++)
            colidx[i]++;
    }

/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
    static inline int transpose_idx(int idx, int dim1, int dim2) {
        return idx % dim1 * dim2 + idx / dim1;
    }

    int CSR::getBandwidth() const {
        int base = getBase();
        const offset_type *rowptr = this->rowptr - base;
        const int *colidx = this->colidx - base;

        int bw = INT_MIN;
#pragma omp parallel reduction(max:bw)
        {
            offset_type iBegin, iEnd;
            getLoadBalancedPartition(&iBegin, &iEnd, rowptr + base, m);
            iBegin += base;
            iEnd += base;

            for (auto i = iBegin; i < iEnd; ++i) {
                for (auto j = rowptr[i]; j < rowptr[i + 1]; ++j) {
                    int c = colidx[j];
                    int temp = c - i;
                    if (temp < 0) temp = -temp;
                    bw = max(temp, bw);
                }
            }
        } // omp parallel

        return bw;
    }

    double CSR::getAverageWidth(bool sorted /*= false*/) const {
        int base = getBase();
        const offset_type *rowptr = this->rowptr - base;
        const int *colidx = this->colidx - base;

        unsigned long long total_width = 0;
        if (sorted) {
#pragma omp parallel for reduction(+:total_width)
            for (int i = base; i < m + base; ++i) {
                if (rowptr[i] == rowptr[i + 1]) continue;

                int width = colidx[rowptr[i + 1] - 1] - colidx[rowptr[i]];
                total_width += width;
            }
        } else {
#pragma omp parallel reduction(+:total_width)
            {
                offset_type iBegin, iEnd;
                getLoadBalancedPartition(&iBegin, &iEnd, rowptr + base, m);
                iBegin += base;
                iEnd += base;

                for (auto i = iBegin; i < iEnd; ++i) {
                    if (rowptr[i] == rowptr[i + 1]) continue;

                    int min_row = INT_MAX, max_row = INT_MIN;
                    for (auto j = rowptr[i]; j < rowptr[i + 1]; ++j) {
                        min_row = min(colidx[j], min_row);
                        max_row = max(colidx[j], max_row);
                    }

                    int width = max_row - min_row;
                    total_width += width;
                }
            } // omp parallel
        }

        return (double) total_width / m;
    }

    int CSR::getMaxDegree() const {
        int max_degree = 0;
        for (int i = 0; i < m; ++i) {
            max_degree = std::max<int>(max_degree, rowptr[i + 1] - rowptr[i]);
        }
        return max_degree;
    }
} // namespace SpMP
