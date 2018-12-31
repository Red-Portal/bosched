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
project(bosched)

# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SOURCE_FILES
    ${SOURCE_DIR}/bosched.cpp)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# Build options
set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -g -fPIC -Wall -Wextra -march=native")

message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 17)

## Thread library dependency
find_package(BLAZE REQUIRED)
#find_package(dlib REQUIRED)

set(BLA_STATIC)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(LBPO_SOURCE_DIR ${PROJECT_SOURCE_DIR}/LBPO)
# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${LPBO_SOURCE_DIR}
    ${BLAZE_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME}
    dlib
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES})
