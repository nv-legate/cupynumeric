#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(get_nccl_version ver_var)
  if(NOT NCCL_INCLUDE_DIRS)
    if(NCCL_INCLUDE_DIR)
      set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    else()
      get_target_property(NCCL_INCLUDE_DIRS NCCL::NCCL INTERFACE_INCLUDE_DIRECTORIES)
    endif()
  endif()

  find_file(
    NCCL_INCLUDE_PATH
    NAMES nccl.h
    PATHS ${NCCL_INCLUDE_DIRS}
    REQUIRED
    NO_DEFAULT_PATH
  )
  file(
    STRINGS "${NCCL_INCLUDE_PATH}"
    file_ver
    LIMIT_COUNT 1
    REGEX [=[^#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+]=]
  )
  if(NOT file_ver)
    message(FATAL_ERROR "Could not read NCCL version from ${NCCL_INCLUDE_PATH}")
  endif()
  string(REGEX MATCH [=[[0-9]+]=] file_ver "${file_ver}")

  # The following version number parsing is taken directly from nccl.h
  string(LENGTH "${file_ver}" ver_len)
  if(ver_len EQUAL 4)
    math(EXPR ver_major "${file_ver}/1000")
    math(EXPR ver_minor "(${file_ver}%1000)/100")
  elseif(ver_len EQUAL 5)
    math(EXPR ver_major "${file_ver}/10000")
    math(EXPR ver_minor "(${file_ver}%10000)/100")
  else()
    message(
      FATAL_ERROR
      "Could not parse NCCL version ${file_ver} from ${NCCL_INCLUDE_PATH}"
    )
  endif()
  math(EXPR ver_patch "${file_ver}%100")
  set(${ver_var} "${ver_major}.${ver_minor}.${ver_patch}" PARENT_SCOPE)
endfunction()

function(validate_nccl_version ver)
  # NCCL >=2.28 is required for ncclAlltoAll. If you bump these, also bump
  # the dependency lists.
  # ver_max is an *exclusive* upper bound
  set(ver_min 2.28)
  set(ver_max 2.31)

  if(ver VERSION_LESS ver_min)
    message(FATAL_ERROR "Detected NCCL version ${ver}, but >= ${ver_min} is required")
  endif()
  if(ver VERSION_GREATER_EQUAL ver_max)
    message(FATAL_ERROR "Detected NCCL version ${ver}, but < ${ver_max} is required")
  endif()
  message(STATUS "NCCL version ${ver} meets requirement >= ${ver_min}, < ${ver_max}")
endfunction()

function(find_or_configure_nccl)
  list(APPEND CMAKE_MESSAGE_CONTEXT "nccl")

  if(NOT TARGET NCCL::NCCL)
    # Workaround from #921 where find may fail when mixing conda and system deps
    if(
      CMAKE_SYSTEM_NAME STREQUAL "Linux"
      AND NOT CMAKE_LIBRARY_ARCHITECTURE
      AND EXISTS "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
    )
      message(
        VERBOSE
        "linux system detected\n"
        "CMAKE_LIBRARY_ARCHITECTURE is unset, attempting to deduce it"
      )
      set(CMAKE_LIBRARY_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
    endif()

    rapids_find_generate_module(NCCL HEADER_NAMES nccl.h LIBRARY_NAMES nccl)
    rapids_find_package(NCCL REQUIRED)
  endif()
  if(TARGET NCCL::nccl AND NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL ALIAS NCCL::nccl)
  endif()
  if(NOT NCCL_VERSION)
    get_nccl_version(NCCL_VERSION)
  endif()
  validate_nccl_version(${NCCL_VERSION})
endfunction()

find_or_configure_nccl()
