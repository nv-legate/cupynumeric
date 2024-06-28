#=============================================================================
# Copyright 2022 NVIDIA Corporation
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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif()

if(POLICY CMP0096)
  cmake_policy(SET CMP0096 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0096 NEW)
endif()

if(POLICY CMP0135)
  # make the timestamps of ExternalProject_ADD match the download time
  # https://cmake.org/cmake/help/latest/policy/CMP0135.html
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

if(POLICY CMP0132)
  # Avoid an inconsistency, where cmake would only set the CC/CXX env vars on
  # the first run, but not subsequent ones. This would come up when building
  # TBLIS.
  cmake_policy(SET CMP0132 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0132 NEW)
endif()

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

set(rapids-cmake-version 24.04)
set(rapids-cmake-sha "365322aca32fd6ecd7027f5d7ec7be50b7f3cc2a")
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

set(cunumeric_version 24.06.00)

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
  project(cunumeric VERSION ${cunumeric_version} LANGUAGES C CXX)
  include(cunumeric_cpp.cmake)
else()
  project(
    cunumeric_python
    VERSION ${cunumeric_version}
    LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
              # language to be enabled here. The test project that is built in scikit-build to verify
              # various linking options for the python library is hardcoded to build with C, so until
              # that is fixed we need to keep C.
              C CXX)
  include(cunumeric_python.cmake)
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja")
  function(add_touch_cunumeric_ninja_build_target)
    set(_suf )
    if(SKBUILD)
      set(_suf "_python")
    endif()
    add_custom_target("touch_cunumeric${_suf}_ninja_build" ALL
      COMMAND ${CMAKE_COMMAND} -E touch_nocreate "${CMAKE_CURRENT_BINARY_DIR}/build.ninja"
      COMMENT "touch build.ninja so ninja doesn't re-run CMake on rebuild"
      VERBATIM
    )
    foreach(_dep IN ITEMS cunumeric cunumeric_python
                          legion_core legion_core_python
                          Legion LegionRuntime
                          Realm RealmRuntime
                          Regent)
      if(TARGET ${_dep})
        add_dependencies("touch_cunumeric${_suf}_ninja_build" ${_dep})
      endif()
    endforeach()
  endfunction()
  add_touch_cunumeric_ninja_build_target()
endif()
