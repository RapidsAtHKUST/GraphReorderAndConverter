#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <malloc.h>

#ifndef __INTEL_COMPILER

#include <x86intrin.h>

#endif

#include <omp.h>

#define FREE(x) { if (x) _mm_free(x); x = NULL; }
#define MALLOC(type, len) (type *)_mm_malloc(sizeof(type)*(len), 64)

#include <cstdint>
#include <cassert>
#include <cstdlib>

typedef uint32_t offset_type;

namespace SpMP {

/**
 * Measure CPU frequency by __rdtsc a compute intensive loop.
 */
    double get_cpu_freq();

    template<typename U, typename T>
    bool operator<(const std::pair<U, T> &a, const std::pair<U, T> &b) {
        if (a.first != b.first) {
            return a.first < b.first;
        } else {
            return a.second < b.second;
        }
    }

/**
 * Get a load balanced partition so that each thread can work on
 * the range of begin-end where prefixSum[end] - prefixSum[begin]
 * is similar among threads.
 * For example, prefixSum can be rowptr of a CSR matrix and n can be
 * the number of rows. Then, each thread will work on similar number
 * of non-zeros.
 *
 * @params prefixSum monotonically increasing array with length n + 1
 *
 * @note must be called inside an omp region
 */
    void getLoadBalancedPartition(offset_type *begin, offset_type *end, const offset_type *prefixSum, offset_type n);

/**
 * @return true if perm array is a permutation
 */
    bool isPerm(const int *perm, int n);

    template<class T>
    void copyVector(T *out, const T *in, int len) {
#pragma omp parallel for
        for (int i = 0; i < len; ++i) {
            out[i] = in[i];
        }
    }

#define USE_LARGE_PAGE
#ifdef USE_LARGE_PAGE

#include <sys/mman.h>

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ALIGN_TO_PAGE_SIZE(x) \
(((x) + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE)

#ifndef MAP_HUGETLB
# define MAP_HUGETLB  0x40000
#endif

    inline void *malloc_huge_pages(size_t size) {
// Use 1 extra page to store allocation metadata
// (libhugetlbfs is more efficient in this regard)
        size_t real_size = ALIGN_TO_PAGE_SIZE(size + HUGE_PAGE_SIZE);
        char *ptr = (char *) mmap(NULL, real_size, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS |
                                  MAP_POPULATE | MAP_HUGETLB, -1, 0);
        if (ptr == MAP_FAILED) {
// The mmap() call failed. Try to malloc instead
            posix_memalign((void **) &ptr, 4096, real_size);
            if (ptr == NULL) return NULL;
            real_size = 0;
        }
// Save real_size since mmunmap() requires a size parameter
        *((size_t *) ptr) = real_size;
// Skip the page with metadata
        return ptr + HUGE_PAGE_SIZE;
    }

    inline void free_huge_pages(void *ptr) {
        if (ptr == NULL) return;
// Jump back to the page with metadata
        void *real_ptr = (char *) ptr - HUGE_PAGE_SIZE;
// Read the original allocation size
        size_t real_size = *((size_t *) real_ptr);
        assert(real_size % HUGE_PAGE_SIZE == 0);
        if (real_size != 0)
// The memory was allocated via mmap()
// and must be deallocated via munmap()
            munmap(real_ptr, real_size);
        else
// The memory was allocated via malloc()
// and must be deallocated via free()
            free(real_ptr);
    }

#undef USE_LARGE_PAGE
#endif // USE_LARGE_PAGE

} // namespace SpMP
