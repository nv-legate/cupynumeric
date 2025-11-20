#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(_cupynumeric_download_rapids DEST_PATH)
  set(expected_hash "")
  if(rapids-cmake-version)
    # If the user overrides the version we don't have a hash, so drop any stale file.
    file(REMOVE "${DEST_PATH}")
  else()
    # default
    set(rapids-cmake-version "25.10")
    set(rapids-cmake-sha "84f8cf8386ac56e3f4f9400f44e752345d8c2997")

    # Propagate to callers
    set(rapids-cmake-version "${rapids-cmake-version}" PARENT_SCOPE)
    set(rapids-cmake-sha "${rapids-cmake-sha}" PARENT_SCOPE)

    # Update when bumping rapids-cmake
    set(expected_hash
        EXPECTED_HASH
        SHA256=3ef01fbdcb6d0a38853ca209c780430c4badd8777b5178d1bba33ca13c4a62b9)
  endif()

  set(file_name
      "https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake"
  )
  foreach(idx RANGE 1 5) # inclusive
    file(DOWNLOAD "${file_name}" "${DEST_PATH}" ${expected_hash} STATUS status)

    list(GET status 0 code)
    if(code EQUAL 0)
      return()
    endif()
    message(VERBOSE "Failed to download ${file_name}, retrying.")
    execute_process(COMMAND ${CMAKE_COMMAND} -E sleep "${idx}")
  endforeach()

  file(REMOVE "${DEST_PATH}")
  list(GET status 1 reason)
  message(FATAL_ERROR "Error (${code}) when downloading ${file_name}: ${reason}")
endfunction()

macro(cupynumeric_include_rapids)
  list(APPEND CMAKE_MESSAGE_CONTEXT "include_rapids")

  if(NOT _CUPYNUMERIC_HAS_RAPIDS)
    set(cupynumeric_rapids_file "${CMAKE_CURRENT_BINARY_DIR}/CUPYNUMERIC_RAPIDS.cmake")

    _cupynumeric_download_rapids("${cupynumeric_rapids_file}")
    include("${cupynumeric_rapids_file}")

    unset(cupynumeric_rapids_file)
    set(_CUPYNUMERIC_HAS_RAPIDS ON)
  endif()
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
