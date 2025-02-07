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

endfunction()

find_or_configure_cutensor()
