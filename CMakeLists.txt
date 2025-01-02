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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

cmake_path(SET CUPYNUMERIC_CMAKE_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE STRING "" FORCE)

set(CMAKE_CUDA_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE STRING "" FORCE)

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

set(rapids-cmake-version 24.12)
set(rapids-cmake-sha "4cb2123dc08ef5d47ecdc9cc51c96bea7b5bb79c")
if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake
       ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
endif()
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

set(cupynumeric_version 25.01.00)

# For now we want the optimization flags to match on both normal make and cmake
# builds so we override the cmake defaults here for release, this changes
# -O3 to -O2 and removes -DNDEBUG
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g")

if(NOT SKBUILD)
  project(cupynumeric VERSION ${cupynumeric_version} LANGUAGES C CXX)
  include(cupynumeric_cpp.cmake)
else()
  project(
    cupynumeric_python
    VERSION ${cupynumeric_version}
    LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
              # language to be enabled here. The test project that is built in scikit-build to verify
              # various linking options for the python library is hardcoded to build with C, so until
              # that is fixed we need to keep C.
              C CXX)
  include(cupynumeric_python.cmake)
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja")
  function(add_touch_cupynumeric_ninja_build_target)
    set(_suf )
    if(SKBUILD)
      set(_suf "_python")
    endif()
    add_custom_target("touch_cupynumeric${_suf}_ninja_build" ALL
      COMMAND ${CMAKE_COMMAND} -E touch_nocreate "${CMAKE_CURRENT_BINARY_DIR}/build.ninja"
      COMMENT "touch build.ninja so ninja doesn't re-run CMake on rebuild"
      VERBATIM
    )
    foreach(_dep IN ITEMS cupynumeric cupynumeric_python
                          legate legate_python
                          Legion LegionRuntime
                          Realm RealmRuntime
                          Regent)
      if(TARGET ${_dep})
        add_dependencies("touch_cupynumeric${_suf}_ninja_build" ${_dep})
      endif()
    endforeach()
  endfunction()
  add_touch_cupynumeric_ninja_build_target()
endif()
