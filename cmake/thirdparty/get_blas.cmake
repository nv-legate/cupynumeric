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

function(find_or_configure_OpenBLAS)
  set(oneValueArgs VERSION REPOSITORY BRANCH PINNED_TAG EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(INTERFACE64 OFF)
  set(BLAS_name "OpenBLAS")
  set(BLAS_target "openblas")

  # cuPyNumeric presently requires OpenBLAS
  set(BLA_VENDOR OpenBLAS)

  # TODO: should we find (or build) 64-bit BLAS?
  if(FALSE AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
    set(INTERFACE64 ON)
    set(BLAS_name "OpenBLAS64")
    set(BLAS_target "openblas_64")
    set(BLA_SIZEOF_INTEGER 8)
  endif()

  set(FIND_PKG_ARGS      ${PKG_VERSION}
      GLOBAL_TARGETS     ${BLAS_target}
      BUILD_EXPORT_SET   cupynumeric-exports
      INSTALL_EXPORT_SET cupynumeric-exports)

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
  if(PKG_BRANCH)
    get_cpm_git_args(BLAS_cpm_git_args REPOSITORY ${PKG_REPOSITORY} BRANCH ${PKG_BRANCH})
  else()
    get_cpm_git_args(BLAS_cpm_git_args REPOSITORY ${PKG_REPOSITORY} TAG ${PKG_PINNED_TAG})
  endif()

  cmake_policy(GET CMP0048 CMP0048_orig)
  cmake_policy(GET CMP0054 CMP0054_orig)
  set(CMAKE_POLICY_DEFAULT_CMP0048 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0054 NEW)

  # Force a base CPU type for the openblas build.
  set(_target HASWELL)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
	  set(_target ARMV8)
  endif()

  # BLAS emits a bunch of warnings, -w is the "silence all warnings" flag for clang and
  # GCC
  if(MSVC)
    message(FATAL_ERROR "Don't know how to silence warnings with MSVC")
  endif()
  set(c_flags "${CMAKE_C_FLAGS} -w")
  set(f_flags "${CMAKE_Fortran_FLAGS} -w")
  rapids_cpm_find(BLAS ${FIND_PKG_ARGS}
      CPM_ARGS
        ${BLAS_cpm_git_args}
        EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
        SYSTEM TRUE
        OPTIONS "USE_CUDA 0"
                "C_LAPACK ON"
                "USE_THREAD ON"
                "NUM_PARALLEL 32"
                "BUILD_TESTING OFF"
                "BUILD_WITHOUT_CBLAS OFF"
                "BUILD_WITHOUT_LAPACK OFF"
                "CMAKE_POLICY_VERSION_MINIMUM 3.5"
                "INTERFACE64 ${INTERFACE64}"
                "TARGET ${_target}"
                "USE_OPENMP ${Legion_USE_OpenMP}"
                "CMAKE_C_FLAGS ${c_flags}"
                "CMAKE_Fortran_FLAGS ${f_flags}")

  set(CMAKE_POLICY_DEFAULT_CMP0048 ${CMP0048_orig})
  set(CMAKE_POLICY_DEFAULT_CMP0054 ${CMP0054_orig})

  if(BLAS_ADDED AND (TARGET ${BLAS_target}))

    # Ensure we export the name of the actual target, not an alias target
    get_target_property(BLAS_aliased_target ${BLAS_target} ALIASED_TARGET)
    if(TARGET ${BLAS_aliased_target})
      set(BLAS_target ${BLAS_aliased_target})
    endif()
    # Make an BLAS::BLAS alias target
    if(NOT TARGET BLAS::BLAS)
      add_library(BLAS::BLAS ALIAS ${BLAS_target})
    endif()

    # Set build INTERFACE_INCLUDE_DIRECTORIES appropriately
    get_target_property(BLAS_include_dirs ${BLAS_target} INCLUDE_DIRECTORIES)
    target_include_directories(${BLAS_target}
        PUBLIC $<BUILD_INTERFACE:${BLAS_BINARY_DIR}>
               # lapack[e] etc. include paths
               $<BUILD_INTERFACE:${BLAS_include_dirs}>
               # contains openblas_config.h
               $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
               # contains cblas.h and f77blas.h
               $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
             )

    string(JOIN "\n" code_string
      "if(NOT TARGET BLAS::BLAS)"
      "  add_library(BLAS::BLAS ALIAS ${BLAS_target})"
      "endif()"
    )

    # Generate openblas-config.cmake in build dir
    rapids_export(BUILD BLAS
      VERSION ${PKG_VERSION}
      EXPORT_SET "${BLAS_name}Targets"
      GLOBAL_TARGETS ${BLAS_target}
      FINAL_CODE_BLOCK code_string)

    # Do `CPMFindPackage(BLAS)` in build dir
    rapids_export_package(BUILD BLAS cupynumeric-exports
      VERSION ${PKG_VERSION} GLOBAL_TARGETS ${BLAS_target})

    # Tell cmake where it can find the generated blas-config.cmake
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD BLAS [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cupynumeric-exports)
  endif()
endfunction()

function(cupynumeric_find_existing_blas found)
  # We need to determine which vendor we have, because we want to be able to set the
  # number of OpenMP threads for the library. And since that isn't standard API, each of
  # them have a different way of saying it. For example, openblas_set_num_threads()
  # (OpenBLAS), mkl_set_num_threads() (MKL), BlasSetThreading() (Apples Accelerate).
  #
  # Note also, these names are case-sensitive! These must match exactly what is set by
  # cmake, and if any of them change, we are screwed.
  set(ALL_BLAS_VENDORS Accelerate
    openblas
    mkl
    mkl_em64t
    mkl_ia32
    mkl_intel
    mkl_intel_lp64
    mkl_rt
    blis
    blas
    acml
    acml_mp
    armpl_lp64
    complib_sgimath
    cxml
    dxml
    essl
    f77blas
    flexiblas
    goto2
    scs
    sunperf)

  set(${found} FALSE PARENT_SCOPE)
  foreach(vendor IN LISTS ALL_BLAS_VENDORS)
    set(BLA_VENDOR "${vendor}")
    rapids_find_package(
      BLAS
      BUILD_EXPORT_SET cupynumeric-exports
      INSTALL_EXPORT_SET cupynumeric-exports
    )
    if(BLAS_FOUND)
      string(TOUPPER "${vendor}" VENDOR)
      set(CUPYNUMERIC_BLAS_VENDOR ${VENDOR} PARENT_SCOPE)
      set(${found} TRUE PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

function(find_blas)
  if(BLA_VENDOR)
    rapids_find_package(
      BLAS
      BUILD_EXPORT_SET cupynumeric-exports
      INSTALL_EXPORT_SET cupynumeric-exports
    )
    if(BLAS_FOUND)
      string(TOUPPER "${BLA_VENDOR}" VENDOR)
      set(CUPYNUMERIC_BLAS_VENDOR ${VENDOR} PARENT_SCOPE)
      return()
    endif()
  else()
    cupynumeric_find_existing_blas(BLAS_FOUND)
    if(BLAS_FOUND)
      set(CUPYNUMERIC_BLAS_VENDOR "${CUPYNUMERIC_BLAS_VENDOR}" PARENT_SCOPE)
      return()
    endif()
  endif()

  if(NOT DEFINED cupynumeric_OPENBLAS_VERSION)
    # Before v0.3.18, OpenBLAS's throws CMake errors when configuring
    # Versions after v0.3.23 conflict with Realm's OpenMP runtime
    # see https://github.com/nv-legate/cupynumeric.internal/issues/342
    set(cupynumeric_OPENBLAS_VERSION "0.3.23")
  endif()

  if(NOT DEFINED cupynumeric_OPENBLAS_BRANCH)
    set(cupynumeric_OPENBLAS_BRANCH "")
  endif()

  if(NOT DEFINED cupynumeric_OPENBLAS_TAG)
    set(cupynumeric_OPENBLAS_TAG v${cupynumeric_OPENBLAS_VERSION})
  endif()

  if(NOT DEFINED cupynumeric_OPENBLAS_REPOSITORY)
    set(cupynumeric_OPENBLAS_REPOSITORY https://github.com/xianyi/OpenBLAS.git)
  endif()

  find_or_configure_OpenBLAS(
    VERSION          ${cupynumeric_OPENBLAS_VERSION}
    REPOSITORY       ${cupynumeric_OPENBLAS_REPOSITORY}
    BRANCH           ${cupynumeric_OPENBLAS_BRANCH}
    PINNED_TAG       ${cupynumeric_OPENBLAS_TAG}
    EXCLUDE_FROM_ALL ${cupynumeric_EXCLUDE_OPENBLAS_FROM_ALL}
  )

  set(CUPYNUMERIC_BLAS_VENDOR OPENBLAS PARENT_SCOPE)
endfunction()

find_blas()
