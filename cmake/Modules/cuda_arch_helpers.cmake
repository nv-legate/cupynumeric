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

function(set_cuda_arch_from_names)
  set(cuda_archs "")
  # translate legacy arch names into numbers
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "fermi")
    list(APPEND cuda_archs 20)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "kepler")
    list(APPEND cuda_archs 30)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "k20")
    list(APPEND cuda_archs 35)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "k80")
    list(APPEND cuda_archs 37)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "maxwell")
    list(APPEND cuda_archs 52)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "pascal")
    list(APPEND cuda_archs 60)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "volta")
    list(APPEND cuda_archs 70)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "turing")
    list(APPEND cuda_archs 75)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "ampere")
    list(APPEND cuda_archs 80)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "ada")
    list(APPEND cuda_archs 89)
  endif()
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "hopper")
    list(APPEND cuda_archs 90)
  endif()

  if(cuda_archs)
    list(LENGTH cuda_archs num_archs)
    if(num_archs GREATER 1)
      # A CMake architecture list entry of "80" means to build both compute and sm.
      # What we want is for the newest arch only to build that way, while the rest
      # build only for sm.
      list(POP_BACK cuda_archs latest_arch)
      list(TRANSFORM cuda_archs APPEND "-real")
      list(APPEND cuda_archs ${latest_arch})
    else()
      list(TRANSFORM cuda_archs APPEND "-real")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${cuda_archs} PARENT_SCOPE)
  endif()
endfunction()

function(add_cuda_architecture_defines defs)
  message(VERBOSE "legate: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

  set(_defs ${${defs}})

  macro(add_def_if_arch_enabled arch def)
    if("${arch}" IN_LIST CMAKE_CUDA_ARCHITECTURES OR
      ("${arch}-real" IN_LIST CMAKE_CUDA_ARCHITECTURES) OR
      ("${arch}-virtual" IN_LIST CMAKE_CUDA_ARCHITECTURES))
      list(APPEND _defs ${def})
    endif()
  endmacro()

  add_def_if_arch_enabled("20" "FERMI_ARCH")
  add_def_if_arch_enabled("30" "KEPLER_ARCH")
  add_def_if_arch_enabled("35" "K20_ARCH")
  add_def_if_arch_enabled("37" "K80_ARCH")
  add_def_if_arch_enabled("52" "MAXWELL_ARCH")
  add_def_if_arch_enabled("60" "PASCAL_ARCH")
  add_def_if_arch_enabled("70" "VOLTA_ARCH")
  add_def_if_arch_enabled("75" "TURING_ARCH")
  add_def_if_arch_enabled("80" "AMPERE_ARCH")
  add_def_if_arch_enabled("89" "ADA_ARCH")
  add_def_if_arch_enabled("90" "HOPPER_ARCH")

  set(${defs} ${_defs} PARENT_SCOPE)
endfunction()
