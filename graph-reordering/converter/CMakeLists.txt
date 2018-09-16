find_package(TBB)
if (TBB_FOUND)
    add_compile_options(-DTBB=1)
    message("${TBB_VERSION}, ${TBB_INCLUDE_DIRS}, ${TBB_LIBRARIES}")
    include_directories(${TBB_INCLUDE_DIRS})
    link_libraries(${TBB_LIBRARIES})
endif ()

find_package(OpenMP)
add_definitions(-DUSE_LOG)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

if (KNL)
    link_libraries(${MEMKIND_LIBRARIES})
endif ()

add_executable(pscan_input_to_edge_list pscan_input_to_edge_list.cpp pscan_graph.cpp pscan_graph.h
        ../utils/log.cpp ../utils/log.h output_util.cpp output_util.h)


add_executable(pscan_input_to_edge_list_txt pscan_input_to_edge_list_txt.cpp pscan_graph.cpp pscan_graph.h
        ../utils/log.cpp ../utils/log.h output_util.cpp output_util.h)

add_executable(edge_list_head edge_list_head.cpp ../utils/log.cpp pscan_graph.cpp)

add_executable(reordering_pscan_input reordering_pscan_input.cpp ../utils/log.cpp pscan_graph.cpp output_util.cpp output_util.h)

add_executable(edge_list_to_csr_performance_measure edge_list_to_csr_performance_measure.cpp ../utils/log.cpp pscan_graph.cpp output_util.cpp)
add_executable(edge_list_to_csr_performance_measure_deg_descending edge_list_to_csr_performance_measure_deg_descending.cpp
        ../utils/log.cpp pscan_graph.cpp output_util.cpp)

if (KNL)
    add_executable(edge_list_to_csr_performance_measure_hbw edge_list_to_csr_performance_measure.cpp ../utils/log.cpp pscan_graph.cpp output_util.cpp)
    target_compile_definitions(edge_list_to_csr_performance_measure_hbw PRIVATE MEM_KIND=1)

    add_executable(edge_list_to_csr_performance_measure_deg_descending_hbw edge_list_to_csr_performance_measure_deg_descending.cpp
            ../utils/log.cpp pscan_graph.cpp output_util.cpp)
    target_compile_definitions(edge_list_to_csr_performance_measure_deg_descending_hbw PRIVATE MEM_KIND=1)

endif ()