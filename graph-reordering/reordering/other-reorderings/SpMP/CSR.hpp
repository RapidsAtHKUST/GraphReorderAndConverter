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

#pragma once

#include <cstdlib>
#include <vector>
#include <string>

#include "MemoryPool.hpp"
#include "Utils.hpp"

namespace SpMP {
    class CSR {
    public:
        int m;
        int n;
        offset_type *rowptr;
        int *colidx;

        // Following two constructors will make CSR own the data
        explicit CSR(const char *file, int base = 0, bool forceSymmetric = false, int pad = 1);

        CSR(const CSR &A);

        // Following constructor will make CSR does not own the data
        CSR(int m, int n, offset_type *rowptr, int *colidx);

        ~CSR();

        /**
         * Load PETSc bin format
         */
        void loadBin(const char *fileName, int base = 0);

    public:
        /**
         * get reverse Cuthill Mckee permutation that tends to reduce the bandwidth
         *
         * @note only works for a symmetric matrix
         *
         * @param pseudoDiameterSourceSelection true to use heurstic of using a source
         *                                      in a pseudo diameter.
         *                                      Further reduce diameter at the expense
         *                                      of more time on finding permutation.
         */
        void getRCMPermutation(int *perm, int *inversePerm, bool pseudoDiameterSourceSelection = true);

        void getBFSPermutation(int *perm, int *inversePerm);

    public:
        void make0BasedIndexing();

        void make1BasedIndexing();

        void alloc(int m, int nnz);

        void dealloc();

        bool useMemoryPool_() const;

    public:
        int getBandwidth() const;

        double getAverageWidth(bool sorted = false) const;

        int getMaxDegree() const;

        offset_type getNnz() const { return rowptr[m] - getBase(); }

        offset_type getBase() const { return rowptr[0]; }

    private:
        bool ownData_;
    }; // CSR
} // namespace SpMP
