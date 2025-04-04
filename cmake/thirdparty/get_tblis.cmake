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

function(find_or_configure_tblis)
  set(oneValueArgs VERSION REPOSITORY BRANCH EXCLUDE_FROM_ALL USE_OPENMP)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
  get_cpm_git_args(tblis_cpm_git_args REPOSITORY ${PKG_REPOSITORY} BRANCH ${PKG_BRANCH})

  set(lib_suffix "")
  if(BUILD_SHARED_LIBS)
    add_library(tblis SHARED IMPORTED GLOBAL)
    set(lib_suffix "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  else()
    add_library(tblis STATIC IMPORTED GLOBAL)
    set(lib_suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  endif()

  rapids_find_generate_module(tblis
                HEADER_NAMES "tblis/tblis.h"
                LIBRARY_NAMES "libtblis${lib_suffix}"
                NO_CONFIG
                BUILD_EXPORT_SET   cunumeric-exports
                INSTALL_EXPORT_SET cunumeric-exports
  )

  rapids_cpm_find(tblis ${PKG_VERSION}
      GLOBAL_TARGETS    tblis::tblis
      BUILD_EXPORT_SET   cunumeric-exports
      INSTALL_EXPORT_SET cunumeric-exports
      CPM_ARGS
        ${tblis_cpm_git_args}
        EXCLUDE_FROM_ALL  ${PKG_EXCLUDE_FROM_ALL}
  )


  set(should_build_tblis OFF)
  if(tblis_ADDED AND
    (NOT EXISTS "${tblis_BINARY_DIR}/include") OR
    (NOT EXISTS "${tblis_BINARY_DIR}/lib/libtci${lib_suffix}") OR
    (NOT EXISTS "${tblis_BINARY_DIR}/lib/libtblis${lib_suffix}"))
    set(should_build_tblis ${tblis_ADDED})
  endif()

  message(VERBOSE "tblis_ADDED: ${tblis_ADDED}")
  message(VERBOSE "tblis_SOURCE_DIR: ${tblis_SOURCE_DIR}")
  message(VERBOSE "tblis_BINARY_DIR: ${tblis_BINARY_DIR}")
  message(VERBOSE "should_build_tblis: ${should_build_tblis}")

  if(should_build_tblis)

    set(should_configure_tblis ON)
    if (EXISTS "${tblis_SOURCE_DIR}/Makefile")
      set(should_configure_tblis OFF)
    endif()

    message(VERBOSE "should_configure_tblis: ${should_configure_tblis}")

    # Configure tblis
    if (should_configure_tblis)
      set(tblis_thread_model "--disable-thread-model")
      if(PKG_USE_OPENMP)
        set(tblis_thread_model "--enable-thread-model=openmp")
      endif()

      # Use ENV{CC/CXX} to tell TBLIS to use the same compilers as the
      # rest of the build.
      # TODO: Consider doing the same for CMAKE_C/CXX_FLAGS
      set(CC_ORIG "$ENV{CC}")
      set(CXX_ORIG "$ENV{CXX}")
      set(_CC "${CMAKE_C_COMPILER}")
      set(_CXX "${CMAKE_CXX_COMPILER}")

      # Use the caching compiler (if provided) to speed up tblis builds
      if(CMAKE_C_COMPILER_LAUNCHER)
        set(_CC "${CMAKE_C_COMPILER_LAUNCHER} ${_CC}")
      endif()
      if(CMAKE_CXX_COMPILER_LAUNCHER)
        set(_CXX "${CMAKE_CXX_COMPILER_LAUNCHER} ${_CXX}")
      endif()

      set(ENV{CC} "${_CC}")
      set(ENV{CXX} "${_CXX}")
      message(VERBOSE "cunumeric: ENV{CC}=\"$ENV{CC}\"")
      message(VERBOSE "cunumeric: ENV{CXX}=\"$ENV{CXX}\"")

      set(tblis_verbosity "--enable-silent-rules")
      if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.25")
        cmake_language(GET_MESSAGE_LOG_LEVEL log_level)
        if(${log_level} STREQUAL "VERBOSE" OR
           ${log_level} STREQUAL "DEBUG" OR
           ${log_level} STREQUAL "TRACE")
             set(tblis_verbosity "--disable-silent-rules")
        endif()
      endif()

      execute_process(
        COMMAND ./configure
          ${tblis_thread_model}
          ${tblis_verbosity}
          --disable-option-checking
          --with-label-type=int32_t
          --with-length-type=int64_t
          --with-stride-type=int64_t
          --prefix=${tblis_BINARY_DIR}
          --disable-dependency-tracking
        WORKING_DIRECTORY      "${tblis_SOURCE_DIR}"
        COMMAND_ECHO           STDOUT
        COMMAND_ERROR_IS_FATAL ANY
        ECHO_ERROR_VARIABLE
        ECHO_OUTPUT_VARIABLE)

      # Reset ENV{CC/CXX}
      set(ENV{CC} "${CC_ORIG}")
      set(ENV{CXX} "${CXX_ORIG}")
    endif()

    # Build tblis into ${tblis_BINARY_DIR}
    add_custom_command(
      OUTPUT
        "${tblis_BINARY_DIR}/lib/libtci${lib_suffix}"
        "${tblis_BINARY_DIR}/lib/libtblis${lib_suffix}"
      COMMENT           "Building tblis"
      COMMAND           make -j${CMAKE_BUILD_PARALLEL_LEVEL} install
      DEPENDS           "${tblis_SOURCE_DIR}/Makefile"
      WORKING_DIRECTORY "${tblis_SOURCE_DIR}"
      USES_TERMINAL
      VERBATIM
    )

    # Makes `target_include_directories()` below work
    file(MAKE_DIRECTORY "${tblis_BINARY_DIR}/include")
  endif()

  if (tblis_ADDED)
    # We need to make the tblis target here since we
    # did not find an external package.
    add_custom_target(tblis_build ALL
      DEPENDS "${tblis_BINARY_DIR}/lib/libtci${lib_suffix}"
              "${tblis_BINARY_DIR}/lib/libtblis${lib_suffix}")

    add_dependencies(tblis tblis_build)

    add_library(tblis::tblis ALIAS tblis)
    target_include_directories(tblis INTERFACE "${tblis_BINARY_DIR}/include")
    set_target_properties(tblis
      PROPERTIES BUILD_RPATH                         "\$ORIGIN"
                 INSTALL_RPATH                       "\$ORIGIN"
                 IMPORTED_SONAME                     tblis
                 IMPORTED_LOCATION                   "${tblis_BINARY_DIR}/lib/libtblis${lib_suffix}"
                 INSTALL_REMOVE_ENVIRONMENT_RPATH    ON
                 INTERFACE_POSITION_INDEPENDENT_CODE ON)
  endif()

  set(tblis_BINARY_DIR ${tblis_BINARY_DIR} PARENT_SCOPE)
  set(cunumeric_INSTALL_TBLIS ${should_build_tblis} PARENT_SCOPE)
endfunction()

if(NOT DEFINED cunumeric_TBLIS_BRANCH)
  set(cunumeric_TBLIS_BRANCH arm-build)
endif()

if(NOT DEFINED cunumeric_TBLIS_REPOSITORY)
  set(cunumeric_TBLIS_REPOSITORY https://github.com/nv-legate/tblis.git)
endif()

find_or_configure_tblis(VERSION          1.2.0
                        REPOSITORY       ${cunumeric_TBLIS_REPOSITORY}
                        BRANCH           ${cunumeric_TBLIS_BRANCH}
                        EXCLUDE_FROM_ALL ${cunumeric_EXCLUDE_TBLIS_FROM_ALL}
                        USE_OPENMP       ${Legion_USE_OpenMP}
)
