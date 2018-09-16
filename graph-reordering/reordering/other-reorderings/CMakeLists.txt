find_package(Boost REQUIRED program_options)
find_package(OpenMP)
find_package(Numa)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

#### RCM (reverse Cuthill-McKee (RCM))
add_executable(rcm_order SpMP/reordering_test.cpp
        SpMP/Utils.cpp SpMP/MemoryPool.cpp
        SpMP/synk/barrier.cpp SpMP/synk/barrier.hpp SpMP/synk/synk.cpp SpMP/synk/synk.hpp SpMP/synk/cpuid.h SpMP/synk/atomic.h
        SpMP/CSR.cpp SpMP/reordering/RCM.cpp SpMP/reordering/ConnectedComponents.cpp SpMP/reordering/BitVector.hpp
        ../../converter/pscan_graph.cpp ../../utils/log.cpp)

target_compile_definitions(rcm_order PRIVATE USE_LOG=1)

#### gorder cache
add_executable(gorder_cache Gorder/Graph.cpp Gorder/Graph.h Gorder/UnitHeap.cpp Gorder/UnitHeap.h Gorder/Util.cpp Gorder/Util.h Gorder/main.cpp)

#### rabbit order
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-std=c++14 HAVE_FLAG_STD_CXX14)
if (HAVE_FLAG_STD_CXX14)
    message("can compile rabbit-order")
    # Have -std=c++14, use it
    set(CMAKE_CXX_STANDARD 14)

    add_executable(rabbit_order rabbit_order/demo/reorder.cc rabbit_order/rabbit_order.hpp rabbit_order/demo/edge_list.hpp)
    target_include_directories(rabbit_order PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(rabbit_order PRIVATE ${NUMA_LIBRARY})
else ()
    message("cannot compile rabbit order, need g++ 4.9.2+")
endif ()