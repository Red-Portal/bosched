# MIT License

# Copyright (c) 2017 Red-Portal

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.8)
project(bench)

# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SOURCE_FILES
    ${SOURCE_DIR}/uniform.cpp
    
    ${SOURCE_DIR}/bias2.cpp
    ${SOURCE_DIR}/bias3.cpp
    )

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

add_executable(uniform ${SOURCE_DIR}/uniform.cpp)
add_executable(bias1   ${SOURCE_DIR}/bias1.cpp)
add_executable(bias2   ${SOURCE_DIR}/bias2.cpp)
add_executable(bias3   ${SOURCE_DIR}/bias3.cpp)

# Build options
set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -g -fPIC -Wall -Wextra -march=native -fopenmp")

message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(uniform
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 17)
set_target_properties(bias1
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 17)
set_target_properties(bias2
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 17)
set_target_properties(bias3
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 17)

## Thread library dependency
find_package(BLAZE REQUIRED)
#find_package(dlib REQUIRED)

#set(BLA_STATIC ON)
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED COMPONENTS program_options thread)

# Include directories
target_include_directories(uniform
    PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${BLAZE_INCLUDE_DIR})

target_include_directories(bias1
    PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${BLAZE_INCLUDE_DIR})

target_include_directories(bias2
    PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${BLAZE_INCLUDE_DIR})

target_include_directories(bias3
    PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${BLAZE_INCLUDE_DIR})

target_link_libraries(uniform
    gomp
    openblas
    lapack
    bosched
    ${Boost_LIBRARIES}
    ${LAPACK_LIBRARIES})

target_link_libraries(bias1
    gomp
    openblas
    lapack
    bosched
    ${Boost_LIBRARIES}
    ${LAPACK_LIBRARIES})

target_link_libraries(bias2
    gomp
    openblas
    lapack
    bosched
    ${Boost_LIBRARIES}
    ${LAPACK_LIBRARIES})

target_link_libraries(bias3
    gomp
    openblas
    lapack
    bosched
    ${Boost_LIBRARIES}
    ${LAPACK_LIBRARIES})
