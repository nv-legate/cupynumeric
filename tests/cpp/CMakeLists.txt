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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cpp_tests VERSION 0.1 LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  # To catch people trying to build the tests from within tests/cpp instead of top-level
  message(FATAL_ERROR "Error: Tests can only be built as part of the main library build. Please re-run cmake from top-level directory (\${CMAKE_SOURCE_DIR}) with -Dcupynumeric_BUILD_TESTS=ON"
  )
endif()

if(Legion_USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
endif()

include(rapids-test)

rapids_test_init()

include(${rapids-cmake-dir}/cpm/gtest.cmake)

# BUILD_EXPORT_SET and INSTALL_EXPORT_SET are crucial, otherwise gtest does not get
# installed
rapids_cpm_gtest(BUILD_EXPORT_SET cupynumeric-exports
                 INSTALL_EXPORT_SET cupynumeric-exports)

file(GLOB main_SRC ${PROJECT_SOURCE_DIR}/main.cc)
file(GLOB integration_SRC ${PROJECT_SOURCE_DIR}/integration/*.cc)

if(Legion_USE_CUDA)
  file(GLOB integration_GPU_SRC ${PROJECT_SOURCE_DIR}/integration/*.cu)
  list(APPEND integration_SRC ${integration_GPU_SRC})
endif()

add_executable(cpp_tests ${main_SRC} ${tasks_SRC} ${integration_SRC} ${unit_SRC})

target_link_libraries(cpp_tests PRIVATE legate::legate cupynumeric::cupynumeric GTest::gtest GTest::gmock)
if(Legion_USE_CUDA)
  target_link_libraries(cpp_tests PRIVATE NCCL::NCCL)
endif()

if(Legion_USE_CUDA)
  set(num_gpus 1)
else()
  set(num_gpus 0)
endif()

rapids_test_add(
  NAME cpp_tests
  COMMAND cpp_tests
  GPUS ${num_gpus}
  PERCENT 30
  INSTALL_COMPONENT_SET testing
)

include(GNUInstallDirs)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing
                                DESTINATION ${CMAKE_INSTALL_BINDIR} INCLUDE_IN_ALL)
