#=============================================================================
# Copyright 2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.26.4)

project(cupynumeric-python VERSION 25.08.00 LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_INSTALL_LIBDIR lib64)

if(APPLE)
  # Workaround the way CCCL uses CMake variables to configure the default Thrust.
  set(CCCL_THRUST_DEVICE_SYSTEM "CPP" CACHE STRING "Use CPP")
  # Using native toolchain, no OpenMP there.
  set(Legion_USE_OpenMP OFF)
else()
  # We want to force the pip wheels to build and ship their own openblas
  set(BLA_VENDOR "OpenBLAS")
endif()

set(CUPYNUMERIC_BUILD_PIP_WHEELS ON)

add_subdirectory(../../../.. cupynumeric-all)

if(APPLE)
  set(rpaths "@loader_path/../../legate/lib64")
else()
  set(rpaths
    "$ORIGIN/../../legate/lib64"
    "$ORIGIN/../../cutensor/lib"
    "$ORIGIN/../../nvidia/cublas/lib"
    "$ORIGIN/../../nvidia/cufft/lib"
    "$ORIGIN/../../nvidia/cusolver/lib"
    "$ORIGIN/../../nvidia/cusparse/lib"
    "$ORIGIN/../../nvidia/nvjitlink/lib"
  )
endif()
set_property(
  TARGET cupynumeric
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
