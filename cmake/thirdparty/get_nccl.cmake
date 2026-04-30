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

function(validate_cupynumeric_nccl_version)

    if(NOT cupynumeric_nccl_include_dir)
        get_target_property(nccl_include_dirs NCCL::NCCL INTERFACE_INCLUDE_DIRECTORIES)

        set(candidate_incdirs)
        foreach(potential_dir IN LISTS nccl_include_dirs)
            string(GENEX_STRIP "${potential_dir}" stripped_dir)
            list(APPEND candidate_incdirs "${stripped_dir}")
        endforeach()

        find_path(cupynumeric_nccl_include_dir
            NAMES nccl.h
            PATH_SUFFIXES nccl
            HINTS ${candidate_incdirs}
            DOC "Path containing nccl.h"
        )
    endif()

    if(NOT cupynumeric_nccl_include_dir)
        message(FATAL_ERROR
            "Could not find nccl.h in any include directory for "
            "NCCL version validation. Searched: ${candidate_incdirs}"
        )
    endif()

    set(nccl_header_path "${cupynumeric_nccl_include_dir}/nccl.h")

    file(STRINGS "${nccl_header_path}" maj_line
        LIMIT_COUNT 1
        REGEX [=[^#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+]=]
    )
    file(STRINGS "${nccl_header_path}" min_line
        LIMIT_COUNT 1
        REGEX [=[^#define[ \t]+NCCL_MINOR[ \t]+[0-9]+]=]
    )

    string(REGEX MATCH [=[[0-9]+]=] nccl_major "${maj_line}")
    string(REGEX MATCH [=[[0-9]+]=] nccl_minor "${min_line}")

    if(nccl_major STREQUAL "" OR nccl_minor STREQUAL "")
        message(FATAL_ERROR "Could not read NCCL version from ${nccl_header_path}")
    endif()

    set(nccl_version "${nccl_major}.${nccl_minor}")
    # NCCL >=2.28 is required for ncclAlltoAll. If you bump these, also bump
    # the dependency lists.
    set(minimum_nccl_version 2.28)
    set(unsupported_nccl_version 2.30)

    if("${nccl_version}" VERSION_LESS "${minimum_nccl_version}")
        message(FATAL_ERROR
            "Detected NCCL version ${nccl_version}, but "
            "version ${minimum_nccl_version} or newer is required."
        )
    endif()

    if("${nccl_version}" VERSION_GREATER_EQUAL "${unsupported_nccl_version}")
        message(FATAL_ERROR
            "Detected NCCL version ${nccl_version}, but "
            "version ${unsupported_nccl_version} or newer is not supported."
        )
    endif()

    message(STATUS
        "NCCL version ${nccl_version} meets requirement "
        ">= ${minimum_nccl_version}, < ${unsupported_nccl_version}"
    )

endfunction()

function(find_or_configure_nccl)

    if(TARGET NCCL::NCCL)
        validate_cupynumeric_nccl_version()
        return()
    endif()

    rapids_find_generate_module(NCCL
        HEADER_NAMES  nccl.h
        LIBRARY_NAMES nccl
    )

    # Currently NCCL has no CMake build-system so we require
    # it built and installed on the machine already
    rapids_find_package(NCCL REQUIRED)
    validate_cupynumeric_nccl_version()

endfunction()

find_or_configure_nccl()
