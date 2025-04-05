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

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuNumeric shared libraries" ON)
option(cunumeric_EXCLUDE_TBLIS_FROM_ALL "Exclude tblis targets from cuNumeric's 'all' target" OFF)
option(cunumeric_EXCLUDE_OPENBLAS_FROM_ALL "Exclude OpenBLAS targets from cuNumeric's 'all' target" OFF)
option(cunumeric_EXCLUDE_LEGATE_CORE_FROM_ALL "Exclude legate.core targets from cuNumeric's 'all' target" OFF)

##############################################################################
# - Project definition -------------------------------------------------------

# Write the version header
rapids_cmake_write_version_file(include/cunumeric/version_config.hpp)

# Needed to integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - Build Type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

##############################################################################
# - conda environment --------------------------------------------------------

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# We're building python extension libraries, which must always be installed
# under lib/, even if the system normally uses lib64/. Rapids-cmake currently
# doesn't realize this when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

##############################################################################
# - Dependencies -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json)

find_package(OpenMP)

option(Legion_USE_CUDA "Use CUDA" ON)
option(Legion_USE_OpenMP "Use OpenMP" ${OpenMP_FOUND})
option(Legion_BOUNDS_CHECKS "Build cuNumeric with bounds checks (expensive)" OFF)

###
# If we find legate.core already configured on the system, it will report
# whether it was compiled with bounds checking (Legion_BOUNDS_CHECKS),
# CUDA (Legion_USE_CUDA), and OpenMP (Legion_USE_OpenMP).
#
# We use the same variables as legate.core because we want to enable/disable
# each of these features based on how legate.core was configured (it doesn't
# make sense to build cuNumeric's CUDA bindings if legate.core wasn't built
# with CUDA support).
###
include(cmake/thirdparty/get_legate_core.cmake)

if(Legion_USE_CUDA)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cuda_arch_helpers.cmake)
  # Needs to run before `rapids_cuda_init_architectures`
  set_cuda_arch_from_names()
  # Needs to run before `enable_language(CUDA)`
  rapids_cuda_init_architectures(cunumeric)
  enable_language(CUDA)
  # Since cunumeric only enables CUDA optionally we need to manually include
  # the file that rapids_cuda_init_architectures relies on `project` calling
  if(CMAKE_PROJECT_cunumeric_INCLUDE)
    include("${CMAKE_PROJECT_cunumeric_INCLUDE}")
  endif()

  # Must come after enable_language(CUDA)
  # Use `-isystem <path>` instead of `-isystem=<path>`
  # because the former works with clangd intellisense
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET cunumeric-exports
    INSTALL_EXPORT_SET cunumeric-exports
  )

  include(cmake/thirdparty/get_nccl.cmake)
  include(cmake/thirdparty/get_cutensor.cmake)
endif()

include(cmake/thirdparty/get_openblas.cmake)

include(cmake/thirdparty/get_tblis.cmake)

##############################################################################
# - cuNumeric ----------------------------------------------------------------

set(cunumeric_SOURCES "")
set(cunumeric_CXX_DEFS "")
set(cunumeric_CUDA_DEFS "")
set(cunumeric_CXX_OPTIONS "")
set(cunumeric_CUDA_OPTIONS "")

include(cmake/Modules/set_cpu_arch_flags.cmake)
set_cpu_arch_flags(cunumeric_CXX_OPTIONS)

# Add `src/cunumeric.mk` sources
list(APPEND cunumeric_SOURCES
  src/cunumeric/ternary/where.cc
  src/cunumeric/scan/scan_global.cc
  src/cunumeric/scan/scan_local.cc
  src/cunumeric/binary/binary_op.cc
  src/cunumeric/binary/binary_red.cc
  src/cunumeric/bits/packbits.cc
  src/cunumeric/bits/unpackbits.cc
  src/cunumeric/unary/scalar_unary_red.cc
  src/cunumeric/unary/unary_op.cc
  src/cunumeric/unary/unary_red.cc
  src/cunumeric/unary/convert.cc
  src/cunumeric/nullary/arange.cc
  src/cunumeric/nullary/eye.cc
  src/cunumeric/nullary/fill.cc
  src/cunumeric/nullary/window.cc
  src/cunumeric/index/advanced_indexing.cc
  src/cunumeric/index/choose.cc
  src/cunumeric/index/putmask.cc
  src/cunumeric/index/repeat.cc
  src/cunumeric/index/select.cc
  src/cunumeric/index/wrap.cc
  src/cunumeric/index/zip.cc
  src/cunumeric/item/read.cc
  src/cunumeric/item/write.cc
  src/cunumeric/matrix/batched_cholesky.cc
  src/cunumeric/matrix/contract.cc
  src/cunumeric/matrix/diag.cc
  src/cunumeric/matrix/gemm.cc
  src/cunumeric/matrix/matmul.cc
  src/cunumeric/matrix/matvecmul.cc
  src/cunumeric/matrix/dot.cc
  src/cunumeric/matrix/potrf.cc
  src/cunumeric/matrix/solve.cc
  src/cunumeric/matrix/syrk.cc
  src/cunumeric/matrix/tile.cc
  src/cunumeric/matrix/transpose.cc
  src/cunumeric/matrix/trilu.cc
  src/cunumeric/matrix/trsm.cc
  src/cunumeric/matrix/util.cc
  src/cunumeric/random/rand.cc
  src/cunumeric/search/argwhere.cc
  src/cunumeric/search/nonzero.cc
  src/cunumeric/set/unique.cc
  src/cunumeric/set/unique_reduce.cc
  src/cunumeric/set/unzip_indices.cc
  src/cunumeric/stat/bincount.cc
  src/cunumeric/convolution/convolve.cc
  src/cunumeric/transform/flip.cc
  src/cunumeric/arg_redop_register.cc
  src/cunumeric/mapper.cc
  src/cunumeric/cephes/chbevl.cc
  src/cunumeric/cephes/i0.cc
  src/cunumeric/stat/histogram.cc
)

if(Legion_USE_OpenMP)
  list(APPEND cunumeric_SOURCES
    src/cunumeric/ternary/where_omp.cc
    src/cunumeric/scan/scan_global_omp.cc
    src/cunumeric/scan/scan_local_omp.cc
    src/cunumeric/binary/binary_op_omp.cc
    src/cunumeric/binary/binary_red_omp.cc
    src/cunumeric/bits/packbits_omp.cc
    src/cunumeric/bits/unpackbits_omp.cc
    src/cunumeric/unary/unary_op_omp.cc
    src/cunumeric/unary/scalar_unary_red_omp.cc
    src/cunumeric/unary/unary_red_omp.cc
    src/cunumeric/unary/convert_omp.cc
    src/cunumeric/nullary/arange_omp.cc
    src/cunumeric/nullary/eye_omp.cc
    src/cunumeric/nullary/fill_omp.cc
    src/cunumeric/nullary/window_omp.cc
    src/cunumeric/index/advanced_indexing_omp.cc
    src/cunumeric/index/choose_omp.cc
    src/cunumeric/index/putmask_omp.cc
    src/cunumeric/index/repeat_omp.cc
    src/cunumeric/index/select_omp.cc
    src/cunumeric/index/wrap_omp.cc
    src/cunumeric/index/zip_omp.cc
    src/cunumeric/matrix/batched_cholesky_omp.cc
    src/cunumeric/matrix/contract_omp.cc
    src/cunumeric/matrix/diag_omp.cc
    src/cunumeric/matrix/gemm_omp.cc
    src/cunumeric/matrix/matmul_omp.cc
    src/cunumeric/matrix/matvecmul_omp.cc
    src/cunumeric/matrix/dot_omp.cc
    src/cunumeric/matrix/potrf_omp.cc
    src/cunumeric/matrix/solve_omp.cc
    src/cunumeric/matrix/syrk_omp.cc
    src/cunumeric/matrix/tile_omp.cc
    src/cunumeric/matrix/transpose_omp.cc
    src/cunumeric/matrix/trilu_omp.cc
    src/cunumeric/matrix/trsm_omp.cc
    src/cunumeric/random/rand_omp.cc
    src/cunumeric/search/argwhere_omp.cc
    src/cunumeric/search/nonzero_omp.cc
    src/cunumeric/set/unique_omp.cc
    src/cunumeric/set/unique_reduce_omp.cc
    src/cunumeric/set/unzip_indices_omp.cc
    src/cunumeric/stat/bincount_omp.cc
    src/cunumeric/convolution/convolve_omp.cc
    src/cunumeric/transform/flip_omp.cc
    src/cunumeric/stat/histogram_omp.cc
  )
endif()

if(Legion_USE_CUDA)
  list(APPEND cunumeric_SOURCES
    src/cunumeric/ternary/where.cu
    src/cunumeric/scan/scan_global.cu
    src/cunumeric/scan/scan_local.cu
    src/cunumeric/binary/binary_op.cu
    src/cunumeric/binary/binary_red.cu
    src/cunumeric/bits/packbits.cu
    src/cunumeric/bits/unpackbits.cu
    src/cunumeric/unary/scalar_unary_red.cu
    src/cunumeric/unary/unary_red.cu
    src/cunumeric/unary/unary_op.cu
    src/cunumeric/unary/convert.cu
    src/cunumeric/nullary/arange.cu
    src/cunumeric/nullary/eye.cu
    src/cunumeric/nullary/fill.cu
    src/cunumeric/nullary/window.cu
    src/cunumeric/index/advanced_indexing.cu
    src/cunumeric/index/choose.cu
    src/cunumeric/index/putmask.cu
    src/cunumeric/index/repeat.cu
    src/cunumeric/index/select.cu
    src/cunumeric/index/wrap.cu
    src/cunumeric/index/zip.cu
    src/cunumeric/item/read.cu
    src/cunumeric/item/write.cu
    src/cunumeric/matrix/batched_cholesky.cu
    src/cunumeric/matrix/contract.cu
    src/cunumeric/matrix/diag.cu
    src/cunumeric/matrix/gemm.cu
    src/cunumeric/matrix/matmul.cu
    src/cunumeric/matrix/matvecmul.cu
    src/cunumeric/matrix/dot.cu
    src/cunumeric/matrix/potrf.cu
    src/cunumeric/matrix/solve.cu
    src/cunumeric/matrix/syrk.cu
    src/cunumeric/matrix/tile.cu
    src/cunumeric/matrix/transpose.cu
    src/cunumeric/matrix/trilu.cu
    src/cunumeric/matrix/trsm.cu
    src/cunumeric/random/rand.cu
    src/cunumeric/search/argwhere.cu
    src/cunumeric/search/nonzero.cu
    src/cunumeric/set/unique.cu
    src/cunumeric/stat/bincount.cu
    src/cunumeric/convolution/convolve.cu
    src/cunumeric/fft/fft.cu
    src/cunumeric/transform/flip.cu
    src/cunumeric/arg_redop_register.cu
    src/cunumeric/cudalibs.cu
    src/cunumeric/stat/histogram.cu
  )
endif()

# Add `src/cunumeric/sort/sort.mk` sources
list(APPEND cunumeric_SOURCES
  src/cunumeric/sort/sort.cc
  src/cunumeric/sort/searchsorted.cc
)

if(Legion_USE_OpenMP)
  list(APPEND cunumeric_SOURCES
    src/cunumeric/sort/sort_omp.cc
    src/cunumeric/sort/searchsorted_omp.cc
  )
endif()

if(Legion_USE_CUDA)
  list(APPEND cunumeric_SOURCES
    src/cunumeric/sort/sort.cu
    src/cunumeric/sort/searchsorted.cu
    src/cunumeric/sort/cub_sort_bool.cu
    src/cunumeric/sort/cub_sort_int8.cu
    src/cunumeric/sort/cub_sort_int16.cu
    src/cunumeric/sort/cub_sort_int32.cu
    src/cunumeric/sort/cub_sort_int64.cu
    src/cunumeric/sort/cub_sort_uint8.cu
    src/cunumeric/sort/cub_sort_uint16.cu
    src/cunumeric/sort/cub_sort_uint32.cu
    src/cunumeric/sort/cub_sort_uint64.cu
    src/cunumeric/sort/cub_sort_half.cu
    src/cunumeric/sort/cub_sort_float.cu
    src/cunumeric/sort/cub_sort_double.cu
    src/cunumeric/sort/thrust_sort_bool.cu
    src/cunumeric/sort/thrust_sort_int8.cu
    src/cunumeric/sort/thrust_sort_int16.cu
    src/cunumeric/sort/thrust_sort_int32.cu
    src/cunumeric/sort/thrust_sort_int64.cu
    src/cunumeric/sort/thrust_sort_uint8.cu
    src/cunumeric/sort/thrust_sort_uint16.cu
    src/cunumeric/sort/thrust_sort_uint32.cu
    src/cunumeric/sort/thrust_sort_uint64.cu
    src/cunumeric/sort/thrust_sort_half.cu
    src/cunumeric/sort/thrust_sort_float.cu
    src/cunumeric/sort/thrust_sort_double.cu
    src/cunumeric/sort/thrust_sort_complex64.cu
    src/cunumeric/sort/thrust_sort_complex128.cu
  )
endif()

# Add `src/cunumeric/random/random.mk` sources
if(Legion_USE_CUDA OR cunumeric_cuRAND_INCLUDE_DIR)
  list(APPEND cunumeric_SOURCES
    src/cunumeric/random/bitgenerator.cc
    src/cunumeric/random/randutil/generator_host.cc
    src/cunumeric/random/randutil/generator_host_straightforward.cc
    src/cunumeric/random/randutil/generator_host_advanced.cc
  )
  if(Legion_USE_CUDA)
    list(APPEND cunumeric_SOURCES
      src/cunumeric/random/bitgenerator.cu
      src/cunumeric/random/randutil/generator_device.cu
      src/cunumeric/random/randutil/generator_device_straightforward.cu
      src/cunumeric/random/randutil/generator_device_advanced.cu
    )
  endif()
endif()

list(APPEND cunumeric_SOURCES
  # This must always be the last file!
  # It guarantees we do our registration callback
  # only after all task variants are recorded
  src/cunumeric/cunumeric.cc
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND cunumeric_CXX_DEFS DEBUG_CUNUMERIC)
  list(APPEND cunumeric_CUDA_DEFS DEBUG_CUNUMERIC)
endif()

if(Legion_BOUNDS_CHECKS)
  list(APPEND cunumeric_CXX_DEFS BOUNDS_CHECKS)
  list(APPEND cunumeric_CUDA_DEFS BOUNDS_CHECKS)
endif()

list(APPEND cunumeric_CUDA_OPTIONS -Xfatbin=-compress-all)
list(APPEND cunumeric_CUDA_OPTIONS --expt-extended-lambda)
list(APPEND cunumeric_CUDA_OPTIONS --expt-relaxed-constexpr)
list(APPEND cunumeric_CXX_OPTIONS -Wno-deprecated-declarations)
list(APPEND cunumeric_CUDA_OPTIONS -Wno-deprecated-declarations)

add_library(cunumeric ${cunumeric_SOURCES})
add_library(cunumeric::cunumeric ALIAS cunumeric)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(platform_rpath_origin "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(platform_rpath_origin "@loader_path")
endif ()

set_target_properties(cunumeric
           PROPERTIES BUILD_RPATH                         "${platform_rpath_origin}"
                      INSTALL_RPATH                       "${platform_rpath_origin}"
                      CXX_STANDARD                        17
                      CXX_STANDARD_REQUIRED               ON
                      POSITION_INDEPENDENT_CODE           ON
                      INTERFACE_POSITION_INDEPENDENT_CODE ON
                      CUDA_STANDARD                       17
                      CUDA_STANDARD_REQUIRED              ON
                      LIBRARY_OUTPUT_DIRECTORY            lib)

target_link_libraries(cunumeric
   PUBLIC legate::core
          $<TARGET_NAME_IF_EXISTS:NCCL::NCCL>
  PRIVATE BLAS::BLAS
          tblis::tblis
          # Add Conda library and include paths
          $<TARGET_NAME_IF_EXISTS:conda_env>
          $<TARGET_NAME_IF_EXISTS:CUDA::cufft>
          $<TARGET_NAME_IF_EXISTS:CUDA::cublas>
          $<TARGET_NAME_IF_EXISTS:CUDA::cusolver>
          $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
          $<TARGET_NAME_IF_EXISTS:cutensor::cutensor>)

if(NOT Legion_USE_CUDA AND cunumeric_cuRAND_INCLUDE_DIR)
  list(APPEND cunumeric_CXX_DEFS CUNUMERIC_CURAND_FOR_CPU_BUILD)
  target_include_directories(cunumeric PRIVATE ${cunumeric_cuRAND_INCLUDE_DIR})
endif()

# Change THRUST_DEVICE_SYSTEM for `.cpp` files
if(Legion_USE_OpenMP)
  list(APPEND cunumeric_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND cunumeric_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
elseif(NOT Legion_USE_CUDA)
  list(APPEND cunumeric_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND cunumeric_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()

target_compile_options(cunumeric
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${cunumeric_CXX_OPTIONS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${cunumeric_CUDA_OPTIONS}>")

target_compile_definitions(cunumeric
  PUBLIC  "$<$<COMPILE_LANGUAGE:CXX>:${cunumeric_CXX_DEFS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${cunumeric_CUDA_DEFS}>")

target_include_directories(cunumeric
  PRIVATE
    $<BUILD_INTERFACE:${cunumeric_SOURCE_DIR}/src>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

if(Legion_USE_CUDA)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=])

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(cunumeric PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
endif()

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS cunumeric
        DESTINATION ${lib_dir}
        EXPORT cunumeric-exports)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cunumeric/version_config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cunumeric)

if(cunumeric_INSTALL_TBLIS)
  install(DIRECTORY ${tblis_BINARY_DIR}/lib/ DESTINATION ${lib_dir})
  install(DIRECTORY ${tblis_BINARY_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide targets for cuNumeric, an aspiring drop-in replacement for NumPy at scale.

Imported Targets:
  - cunumeric::cunumeric

]=])

string(JOIN "\n" code_string
  "set(Legion_USE_CUDA ${Legion_USE_CUDA})"
  "set(Legion_USE_OpenMP ${Legion_USE_OpenMP})"
  "set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS})"
)

if(DEFINED Legion_USE_Python)
  string(APPEND code_string "\nset(Legion_USE_Python ${Legion_USE_Python})")
endif()

if(DEFINED Legion_NETWORKS)
  string(APPEND code_string "\nset(Legion_NETWORKS ${Legion_NETWORKS})")
endif()

rapids_export(
  INSTALL cunumeric
  EXPORT_SET cunumeric-exports
  GLOBAL_TARGETS cunumeric
  NAMESPACE cunumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD cunumeric
  EXPORT_SET cunumeric-exports
  GLOBAL_TARGETS cunumeric
  NAMESPACE cunumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)
