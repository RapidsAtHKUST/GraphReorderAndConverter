## Compile

* on KNL

```
mkdir -p build & cd build
cmake .. -DKNL=ON
make -j
```

* on CPU

```zsh
mkdir -p build & cd build
cmake .. 
make -j
```

## Usage

### Edge List to CSR Performance Measurement

```zsh
./converter/edge_list_to_csr_performance_measure /nfsshare/share/dataset/snap_friendster tmp.txt
```

* attention: tbb sort is not stable, the same as sequential quick sort in STL, thus the permutation may be **slightly different**

```zsh
./converter/edge_list_to_csr_performance_measure_deg_descending_hbw /home/yche/data/dataset/webgraph_twitter tmp.txt
```

### Rabbit Order (Need C++14)

* reordering to generate the `*.dict`, then convert to corresponding ppscan input formats, codes see [rabbit_order](reordering/other-reorderings/rabbit_order), and [converter/reordering_pscan_input.cpp](converter/reordering_pscan_input.cpp).

```zsh
./reordering/other-reorderings/rabbit_order /nfsshare/share/dataset/small_snap_dblp/undir_edge_list.bin &> /dev/null
./converter/reordering_pscan_input /nfsshare/share/dataset/small_snap_dblp rabbit_order tmp.txt 
```

### RCM/BFS by Intel Lab

* RCM is just BFS with Sorting in each level by degree

```zsh
./reordering/other-reorderings/rcm_order /nfsshare/share/dataset/small_snap_dblp

./converter/reordering_pscan_input /nfsshare/share/dataset/small_snap_dblp bfs tmp.txt 
./converter/reordering_pscan_input /nfsshare/share/dataset/small_snap_dblp rcm_with_src_sel tmp.txt 
./converter/reordering_pscan_input /nfsshare/share/dataset/small_snap_dblp rcm_wo_src_sel tmp.txt 
```