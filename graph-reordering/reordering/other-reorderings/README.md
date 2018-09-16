# GraphOrderingAndCompression

Graph Ordering And Compression

RCM： 

* https://github.com/IntelLabs/SpMP
* https://github.com/tnas/reordering-library/tree/master/Reorderings
* https://github.com/michel94/fhgc-tool

Gorder - Cache

rabbit-order (IPDPS)

## Updates

### SpMP (Parallel BFS/RCM)

* remove reordering-irrelevant things: e.g, `COO`, `GS`, etc.

* to support snap-friendster, change CSR, see [SpMP/CSR.hpp](SpMP/CSR.hpp): 

```cpp
// yche: rowptr, extptr: should be uint32_t to support snap_friendster
```