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

# This is based on the similar function for Legion in the Legate code
function(cupynumeric_maybe_override_legate user_repository user_branch user_version)
  # CPM_ARGS GIT_TAG and GIT_REPOSITORY don't do anything if you have already overridden
  # those options via a rapids_cpm_package_override() call. So we have to conditionally
  # override the defaults (by creating a temporary json file in build dir) only if the
  # user sets them.

  # See https://github.com/rapidsai/rapids-cmake/issues/575. Specifically, this function
  # is pretty much identical to
  # https://github.com/rapidsai/rapids-cmake/issues/575#issuecomment-2045374410.
  cmake_path(SET legate_overrides_json NORMALIZE
             "${CUPYNUMERIC_CMAKE_DIR}/versions.json")
  if(user_repository OR user_branch OR user_version)
    # The user has set either one of these, time to create our cludge.
    file(READ "${legate_overrides_json}" default_legate_json)
    set(new_legate_json "${default_legate_json}")

    if(user_repository)
      string(JSON new_legate_json SET "${new_legate_json}" "packages" "Legate" "git_url"
             "\"${user_repository}\"")
    endif()

    if(user_branch)
      string(JSON new_legate_json SET "${new_legate_json}" "packages" "Legate" "git_tag"
             "\"${user_branch}\"")
    endif()

    if(user_version)
      string(JSON new_legate_json SET "${new_legate_json}" "packages" "Legate" "version"
             "\"${user_version}\"")
    endif()

    string(JSON eq_json EQUAL "${default_legate_json}" "${new_legate_json}")
    if(NOT eq_json)
      cmake_path(SET legate_overrides_json NORMALIZE
                 "${CMAKE_CURRENT_BINARY_DIR}/versions.json")
      file(WRITE "${legate_overrides_json}" "${new_legate_json}")
    endif()
  endif()
  rapids_cpm_package_override("${legate_overrides_json}")
endfunction()

function(find_legate)
  set(options)
  set(oneValueArgs VERSION REPOSITORY BRANCH EXCLUDE_FROM_ALL)
  set(multiValueArgs)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cupynumeric_maybe_override_legate("${PKG_REPOSITORY}" "${PKG_BRANCH}" "${PKG_VERSION}")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(legate version git_repo git_branch shallow exclude_from_all)

  string(REPLACE "00" "0" version "${version}")

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

  rapids_find_package(legate ${version} EXACT CONFIG REQUIRED ${FIND_PKG_ARGS})

  message(STATUS "Found legate@${version}")

  # Workaround for https://github.com/NVIDIA/cccl/issues/5002
  if(Legion_USE_OpenMP)
    rapids_find_package(OpenMP GLOBAL_TARGETS OpenMP::OpenMP_CXX)
    get_target_property(opts OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)
    message(STATUS "openmp interface options ${opts}")
    string(REPLACE [[-Xcompiler=SHELL:]] [[SHELL:-Xcompiler=]] opts "${opts}")
    message(STATUS "openmp interface options after ${opts}")
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS "${opts}")
  endif()

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_CUDA_ARCH ${Legion_CUDA_ARCH} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS} PARENT_SCOPE)

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_CUDA_ARCH=${Legion_CUDA_ARCH}")
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

find_legate(VERSION          ${cupynumeric_LEGATE_VERSION}
            REPOSITORY       ${cupynumeric_LEGATE_REPOSITORY}
            BRANCH           ${cupynumeric_LEGATE_BRANCH}
            EXCLUDE_FROM_ALL ${cupynumeric_EXCLUDE_LEGATE_FROM_ALL}
)
