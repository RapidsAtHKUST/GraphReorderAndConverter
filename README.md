# GraphReorderAndConverter

GraphReorderAndConverter, source codes in [graph-reordering](graph-reordering).

## Edge List to CSR Time

see [edge_list_to_csr_performance_measure_deg_descending.cpp](graph-reordering/converter/edge_list_to_csr_performance_measure_deg_descending.cpp)
and [edge_list_to_csr_performance_measure.cpp](graph-reordering/converter/edge_list_to_csr_performance_measure.cpp) for the OpenMP implementation.

unit: seconds.

platform | statistics
--- | ---
28-core-cpu | [cpu-edge-list-to-csr-performance.md](data-md/lccpu12/09_03_edge-list-to-csr-performance.md)
knl | [knl-edge-list-to-csr-performance.md](data-md/knl/09_03_edge-list-to-csr-performance.md)

### twitter-transform-time on CPU

exec-name | before csr transform time | before sort time | edge list to csr time
--- | --- | --- | ---
edge_list_to_csr_performance_measure | 1.795s | 5.942s | 9.065s
edge_list_to_csr_performance_measure_deg_descending | 1.816s | 5.872s | 11.766s

###  twitter-transform-time on KNL

`hbw` for high-bandwidth-memory (MCDRAM), i.e, allocation `dst` array in the CSR on that multi-channel memory.

exec-name | before csr transform time | before sort time | edge list to csr time
--- | --- | --- | ---
edge_list_to_csr_performance_measure | 2.822s | 6.953s | 10.494s
edge_list_to_csr_performance_measure_hbw | 2.805s | 7.033s | 12.199s
edge_list_to_csr_performance_measure_deg_descending | 2.897s | 7.09s | 14.942s
edge_list_to_csr_performance_measure_deg_descending_hbw | 2.832s | 7.108s | 16.313s
