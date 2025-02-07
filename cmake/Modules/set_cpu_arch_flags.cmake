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

#------------------------------------------------------------------------------#
# Architecture
#------------------------------------------------------------------------------#
if(BUILD_MARCH AND BUILD_MCPU)
  message(FATAL_ERROR "BUILD_MARCH and BUILD_MCPU are incompatible")
endif()

function(set_cpu_arch_flags out_var)
  # Try -march first. On platforms that don't support it, GCC will issue a hard
  # error, so we'll know not to use it. Default is "native", but explicitly
  # setting BUILD_MARCH="" disables use of the flag
  if(BUILD_MARCH)
    set(INTERNAL_BUILD_MARCH ${BUILD_MARCH})
  elseif(NOT DEFINED BUILD_MARCH)
    set(INTERNAL_BUILD_MARCH "native")
  endif()

  set(flags "")

  include(CheckCXXCompilerFlag)
  if(INTERNAL_BUILD_MARCH)
    check_cxx_compiler_flag("-march=${INTERNAL_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
    if(COMPILER_SUPPORTS_MARCH)
      list(APPEND flags "-march=${INTERNAL_BUILD_MARCH}")
    elseif(BUILD_MARCH)
      message(FATAL_ERROR "The flag -march=${INTERNAL_BUILD_MARCH} is not supported by the compiler")
    else()
      unset(INTERNAL_BUILD_MARCH)
    endif()
  endif()

  # Try -mcpu. We do this second because it is deprecated on x86, but
  # GCC won't issue a hard error, so we can't tell if it worked or not.
  if (NOT INTERNAL_BUILD_MARCH AND NOT DEFINED BUILD_MARCH)
    if(BUILD_MCPU)
      set(INTERNAL_BUILD_MCPU ${BUILD_MCPU})
    else()
      set(INTERNAL_BUILD_MCPU "native")
    endif()

    check_cxx_compiler_flag("-mcpu=${INTERNAL_BUILD_MCPU}" COMPILER_SUPPORTS_MCPU)
    if(COMPILER_SUPPORTS_MCPU)
      list(APPEND flags "-mcpu=${INTERNAL_BUILD_MCPU}")
    elseif(BUILD_MCPU)
      message(FATAL_ERROR "The flag -mcpu=${INTERNAL_BUILD_MCPU} is not supported by the compiler")
    else()
      unset(INTERNAL_BUILD_MCPU)
    endif()
  endif()

  # Add flags for Power architectures
  check_cxx_compiler_flag("-maltivec -Werror" COMPILER_SUPPORTS_MALTIVEC)
  if(COMPILER_SUPPORTS_MALTIVEC)
    list(APPEND flags "-maltivec")
  endif()
  check_cxx_compiler_flag("-mabi=altivec -Werror" COMPILER_SUPPORTS_MABI_ALTIVEC)
  if(COMPILER_SUPPORTS_MABI_ALTIVEC)
    list(APPEND flags "-mabi=altivec")
  endif()
  check_cxx_compiler_flag("-mvsx -Werror" COMPILER_SUPPORTS_MVSX)
  if(COMPILER_SUPPORTS_MVSX)
    list(APPEND flags "-mvsx")
  endif()

  set(${out_var} "${flags}" PARENT_SCOPE)
endfunction()

set_cpu_arch_flags(arch_flags)
