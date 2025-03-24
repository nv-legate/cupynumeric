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

project(cupynumeric-python VERSION 25.03.00 LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CUPYNUMERIC_BUILD_PIP_WHEELS ON)

add_subdirectory(../../../.. cupynumeric-all)

set(rpaths
  "$ORIGIN/../../legate/lib64"
  "$ORIGIN/../../cutensor/lib"
  "$ORIGIN/../../nvidia/cublas/lib"
  "$ORIGIN/../../nvidia/cufft/lib"
  "$ORIGIN/../../nvidia/cusolver/lib"
  "$ORIGIN/../../nvidia/cusparse/lib"
  "$ORIGIN/../../nvidia/nvjitlink/lib"
)
set_property(
  TARGET cupynumeric
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
