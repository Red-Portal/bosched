cmake_minimum_required (VERSION 2.8.11)
project(4-SolidBoundaries)
add_executable(4-SolidBoundaries "../lodepng/lodepng.cpp" "Fluid.cpp")
#add_executable(4-SolidBoundaries ${SRC} "Fluid.cpp")
set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -g -fPIC -Wall -Wextra -march=native -pthread -fopenmp")
set_target_properties(4-SolidBoundaries
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 17)
target_link_libraries(4-SolidBoundaries
    PRIVATE
    hdf5
    gomp)
