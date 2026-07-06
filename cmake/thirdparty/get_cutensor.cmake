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

function(find_or_configure_cutensor)

    if(TARGET cutensor::cutensor)
        return()
    endif()

    rapids_find_generate_module(cutensor
        HEADER_NAMES  cutensor.h
        LIBRARY_NAMES cutensor
    )

    # Currently cutensor has no CMake build-system so we require
    # it built and installed on the machine already
    rapids_find_package(cutensor REQUIRED)

    # If cuTENSOR ships no CMake config, the rapids_find_generate_module can't
    # populate cutensor_VERSION. In turn we parse it from the header;
    # leaving it empty if the header can't be matched.
    if(NOT cutensor_VERSION AND cutensor_INCLUDE_DIR AND EXISTS "${cutensor_INCLUDE_DIR}/cutensor.h")
        file(STRINGS "${cutensor_INCLUDE_DIR}/cutensor.h" _cutensor_h REGEX "^#define CUTENSOR_(MAJOR|MINOR|PATCH) ")
        if(_cutensor_h MATCHES "CUTENSOR_MAJOR ([0-9]+)")
            set(_cutensor_major "${CMAKE_MATCH_1}")
        endif()
        if(_cutensor_h MATCHES "CUTENSOR_MINOR ([0-9]+)")
            set(_cutensor_minor "${CMAKE_MATCH_1}")
        endif()
        if(DEFINED _cutensor_major AND DEFINED _cutensor_minor)
            set(cutensor_VERSION "${_cutensor_major}.${_cutensor_minor}")
        endif()
    endif()

    if(NOT cutensor_VERSION)
        message(WARNING "Could not determine the cuTENSOR version; skipping the cuTENSOR>=2.0 version check.")
    elseif(cutensor_VERSION VERSION_LESS "2.0")
        message(FATAL_ERROR "cuTENSOR version ${cutensor_VERSION} is not supported. Please install cuTENSOR 2.0 or later.")
    endif()

endfunction()

find_or_configure_cutensor()
