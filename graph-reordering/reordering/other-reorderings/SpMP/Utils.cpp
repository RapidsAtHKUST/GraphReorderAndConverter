#include <cfloat>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/time.h>

#include <string>
#include <algorithm>

#include "Utils.hpp"

using namespace std;

namespace SpMP {
    double get_cpu_freq() {
        static double freq = DBL_MAX;
        if (DBL_MAX == freq) {
            volatile double a = rand() % 1024, b = rand() % 1024;
            struct timeval tv1, tv2;
            gettimeofday(&tv1, NULL);
            unsigned long long t1 = __rdtsc();
            for (size_t i = 0; i < 1024L * 1024; i++) {
                a += a * b + b / a;
            }
            unsigned long long dt = __rdtsc() - t1;
            gettimeofday(&tv2, NULL);
            freq = dt / ((tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1.e6);
        }

        return freq;
    }

    void getLoadBalancedPartition(offset_type *begin, offset_type *end, const offset_type *prefixSum, offset_type n) {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int base = prefixSum[0];
        int total_work = prefixSum[n] - base;
        int work_per_thread = (total_work + nthreads - 1) / nthreads;

        *begin = tid == 0 ? 0 : lower_bound(prefixSum, prefixSum + n, work_per_thread * tid + base) - prefixSum;
        *end = tid == nthreads - 1 ? n : lower_bound(prefixSum, prefixSum + n, work_per_thread * (tid + 1) + base) -
                                         prefixSum;

        assert(*begin <= *end);
        assert(*begin >= 0 && *begin <= n);
        assert(*end >= 0 && *end <= n);
    }

    bool isPerm(const int *perm, int n) {
        int *temp = new int[n];
        memcpy(temp, perm, sizeof(int) * n);
        sort(temp, temp + n);
        int *last = unique(temp, temp + n);
        if (last != temp + n) {
            memcpy(temp, perm, sizeof(int) * n);
            sort(temp, temp + n);

            for (int i = 0; i < n; ++i) {
                if (temp[i] == i - 1) {
                    printf("%d duplicated\n", i - 1);
                    assert(false);
                    return false;
                } else if (temp[i] != i) {
                    printf("%d missed\n", i);
                    assert(false);
                    return false;
                }
            }
        }
        delete[] temp;
        return true;
    }
} // namespace SpMP
