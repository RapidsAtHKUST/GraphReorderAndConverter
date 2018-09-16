cmake_minimum_required(VERSION 3.6)
project(src)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH}) # To allow CMake to locate our Find*.cmake files

#### some initialization of env
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
add_definitions("-DGIT_SHA1=\"${GIT_SHA1}\"")
message("git version: ${GIT_SHA1}")

include(FindSSE)
if (AVX2_FOUND)
    message("-- Found AVX2")
    add_compile_options(-mavx2)
elseif (AVX_FOUND)
    message("-- Found AVX")
    add_compile_options(-mavx)
elseif (SSE4_1_FOUND)
    message("-- Found sse4.1")
    add_compile_options(-msse4.1)
elseif (SSE2_FOUND)
    message("-- Found sse2")
    add_compile_options(-msse2)
endif ()

find_package(Memkind)
if (MEMKIND_FOUND)
    message("-- Found Memkind: ${MEMKIND_INCLUDE_DIR}, ${MEMKIND_LIBRARIES}")
endif ()
find_package(HWLOC)
if (HWLOC_FOUND)
    message("-- Found HWLOC: ${HWLOC_INCLUDE_DIR}, ${HWLOC_LIBRARIES}")
endif ()
#######################################

include_directories(.)

#find_package(ParMETIS)
find_package(METIS)
find_package(Threads)

if (AVX2_FOUND)
    add_compile_options(-g -O3 -Wall -Wno-unused-result -march=core-avx2 -lmetis -pthread)
else ()
    add_compile_options(-g -O3 -Wall -Wno-unused-result -lmetis -pthread)
endif ()

set(DATA_STRUCTURE_FILES reordering/doublelinked_list.cpp reordering/doublelinked_list.hpp
        reordering/linkedlist_heap.cpp reordering/linkedlist_heap.hpp
        reordering/modified_heap.cpp reordering/modified_heap.hpp)
set(UTIL_FILES utils/util.cpp utils/util.hpp)

##### 1) GRO : graph reordering optimization
if (METIS_FOUND)
    add_executable(reorder ${DATA_STRUCTURE_FILES}
            reordering/porder.cpp reordering/porder.hpp reordering/reorder.cpp
            utils/util.cpp)
    target_link_libraries(reorder ${METIS_LIBRARIES})
endif ()

add_subdirectory(reordering/other-reorderings)
add_subdirectory(converter)