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

function(find_or_configure_legate)
  set(oneValueArgs VERSION REPOSITORY BRANCH EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${PKG_VERSION} legate PKG_VERSION)

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(legate version git_repo git_branch shallow exclude_from_all)

  set(version ${PKG_VERSION})
  set(exclude_from_all ${PKG_EXCLUDE_FROM_ALL})
  if(PKG_BRANCH)
    set(git_branch "${PKG_BRANCH}")
  endif()
  if(PKG_REPOSITORY)
    set(git_repo "${PKG_REPOSITORY}")
  endif()

  set(FIND_PKG_ARGS
      GLOBAL_TARGETS     legate::legate
      BUILD_EXPORT_SET   cupynumeric-exports
      INSTALL_EXPORT_SET cupynumeric-exports)

  # First try to find legate via find_package()
  # so the `Legion_USE_*` variables are visible
  # Use QUIET find by default.
  set(_find_mode QUIET)
  # If legate_DIR/legate_ROOT are defined as something other than empty or NOTFOUND
  # use a REQUIRED find so that the build does not silently download legate.
  if(legate_DIR OR legate_ROOT)
    set(_find_mode REQUIRED)
  endif()
  rapids_find_package(legate ${version} EXACT CONFIG ${_find_mode} ${FIND_PKG_ARGS})

  if(legate_FOUND)
    message(STATUS "CPM: using local package legate@${version}")
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
    get_cpm_git_args(legate_cpm_git_args REPOSITORY ${git_repo} BRANCH ${git_branch})

    message(VERBOSE "cupynumeric: legate version: ${version}")
    message(VERBOSE "cupynumeric: legate git_repo: ${git_repo}")
    message(VERBOSE "cupynumeric: legate git_branch: ${git_branch}")
    message(VERBOSE "cupynumeric: legate exclude_from_all: ${exclude_from_all}")
    message(VERBOSE "cupynumeric: legate legate_cpm_git_args: ${legate_cpm_git_args}")

    rapids_cpm_find(legate ${version} ${FIND_PKG_ARGS}
        CPM_ARGS
          ${legate_cpm_git_args}
          FIND_PACKAGE_ARGUMENTS EXACT
          EXCLUDE_FROM_ALL       ${exclude_from_all}
    )
  endif()

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS} PARENT_SCOPE)

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_USE_OpenMP=${Legion_USE_OpenMP}")
  message(VERBOSE "Legion_BOUNDS_CHECKS=${Legion_BOUNDS_CHECKS}")
endfunction()

foreach(_var IN ITEMS "cupynumeric_LEGATE_VERSION"
                      "cupynumeric_LEGATE_BRANCH"
                      "cupynumeric_LEGATE_REPOSITORY"
                      "cupynumeric_EXCLUDE_LEGATE_FROM_ALL")
  if(DEFINED ${_var})
    # Create a cupynumeric_LEGATE_BRANCH variable in the current scope either from the existing
    # current-scope variable, or the cache variable.
    set(${_var} "${${_var}}")
    # Remove cupynumeric_LEGATE_BRANCH from the CMakeCache.txt. This ensures reconfiguring the same
    # build dir without passing `-Dcupynumeric_LEGATE_BRANCH=` reverts to the value in versions.json
    # instead of reusing the previous `-Dcupynumeric_LEGATE_BRANCH=` value.
    unset(${_var} CACHE)
  endif()
endforeach()

if(NOT DEFINED cupynumeric_LEGATE_VERSION)
  set(cupynumeric_LEGATE_VERSION "${cupynumeric_VERSION}")
endif()

find_or_configure_legate(VERSION          ${cupynumeric_LEGATE_VERSION}
                         REPOSITORY       ${cupynumeric_LEGATE_REPOSITORY}
                         BRANCH           ${cupynumeric_LEGATE_BRANCH}
                         EXCLUDE_FROM_ALL ${cupynumeric_EXCLUDE_LEGATE_FROM_ALL}
)
