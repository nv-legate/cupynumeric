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

##############################################################################
# - User Options  ------------------------------------------------------------

option(FIND_CUPYNUMERIC_CPP "Search for existing cuPyNumeric C++ installations before defaulting to local files"
       OFF)

##############################################################################
# - Dependencies -------------------------------------------------------------

# If the user requested it we attempt to find cupynumeric.
if(FIND_CUPYNUMERIC_CPP)
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${cupynumeric_version} cupynumeric parsed_ver)
  rapids_find_package(cupynumeric ${parsed_ver} EXACT CONFIG
                      GLOBAL_TARGETS     cupynumeric::cupynumeric
                      BUILD_EXPORT_SET   cupynumeric-python-exports
                      INSTALL_EXPORT_SET cupynumeric-python-exports)
else()
  set(cupynumeric_FOUND OFF)
endif()

if(NOT cupynumeric_FOUND)
  set(SKBUILD OFF)
  set(Legion_USE_Python ON)
  set(Legion_BUILD_BINDINGS ON)
  add_subdirectory(. "${CMAKE_CURRENT_SOURCE_DIR}/build")
  set(SKBUILD ON)
endif()

add_library(cupynumeric_python INTERFACE)
add_library(cupynumeric::cupynumeric_python ALIAS cupynumeric_python)
target_link_libraries(cupynumeric_python INTERFACE legate::legate)


get_target_property(cn_is_imported cupynumeric::cupynumeric IMPORTED)
if(cn_is_imported)
    get_target_property(cn_lib cupynumeric::cupynumeric IMPORTED_LOCATION)
    if(NOT cn_lib)
        # Try configuration-specific locations
        foreach(config RELEASE DDEBUG RELWITHDEBINFO MINSIZEREL)
            get_target_property(cn_lib cupynumeric::cupynumeric IMPORTED_LOCATION_${config})
            if(cn_lib)
                break()
            endif()
        endforeach()
    endif()
    if(cn_lib)
      get_filename_component(cn_libdir "${cn_lib}" DIRECTORY)
    endif()
endif()

add_custom_target("generate_install_info_py" ALL
  COMMAND ${CMAKE_COMMAND}
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCUPYNUMERIC_LIBDIR=${cn_libdir}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_install_info_py.cmake"
  COMMENT "Generate install_info.py"
  VERBATIM
)

add_library(cupynumeric_python INTERFACE)
add_library(cupynumeric::cupynumeric_python ALIAS cupynumeric_python)
target_link_libraries(cupynumeric_python INTERFACE legate::legate)

# ############################################################################
# - conda environment --------------------------------------------------------

rapids_cmake_support_conda_env(conda_env)

# We're building python extension libraries, which must always be installed
# under lib/, even if the system normally uses lib64/. Rapids-cmake currently
# doesn't realize this when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

##############################################################################
# - install targets ----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS cupynumeric_python
        DESTINATION ${lib_dir}
        EXPORT cupynumeric-python-exports)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide Python targets for cuPyNumeric, an aspiring drop-in replacement for NumPy at scale.

Imported Targets:
  - cupynumeric::cupynumeric_python

]=])

set(code_string "")

rapids_export(
  INSTALL cupynumeric_python
  EXPORT_SET cupynumeric-python-exports
  GLOBAL_TARGETS cupynumeric_python
  NAMESPACE cupynumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD cupynumeric_python
  EXPORT_SET cupynumeric-python-exports
  GLOBAL_TARGETS cupynumeric_python
  NAMESPACE cupynumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)
