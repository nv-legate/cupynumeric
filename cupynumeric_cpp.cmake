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

option(BUILD_SHARED_LIBS "Build cuPyNumeric shared libraries" ON)
option(cupynumeric_EXCLUDE_TBLIS_FROM_ALL "Exclude tblis targets from cuPyNumeric's 'all' target" OFF)
option(cupynumeric_EXCLUDE_OPENBLAS_FROM_ALL "Exclude OpenBLAS targets from cuPyNumeric's 'all' target" OFF)
option(cupynumeric_EXCLUDE_LEGATE_FROM_ALL "Exclude legate targets from cuPyNumeric's 'all' target" OFF)

##############################################################################
# - Project definition -------------------------------------------------------

# Write the version header
rapids_cmake_write_version_file(include/cupynumeric/version_config.hpp)

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

rapids_find_package(OpenMP GLOBAL_TARGETS OpenMP::OpenMP_CXX)

option(Legion_USE_CUDA "Use CUDA" ON)
option(Legion_USE_OpenMP "Use OpenMP" ${OpenMP_FOUND})
option(Legion_BOUNDS_CHECKS "Build cuPyNumeric with bounds checks (expensive)" OFF)

###
# If we find legate already configured on the system, it will report
# whether it was compiled with bounds checking (Legion_BOUNDS_CHECKS),
# CUDA (Legion_USE_CUDA), and OpenMP (Legion_USE_OpenMP).
#
# We use the same variables as legate because we want to enable/disable
# each of these features based on how legate was configured (it doesn't
# make sense to build cuPyNumeric's CUDA bindings if legate wasn't built
# with CUDA support).
###
include(thirdparty/get_legate)

# Use of DEFINED is deliberate. CMAKE_CUDA_ARCHITECTURES may be OFF which we want to leave
# in place. Legion_CUDA_ARCH is defined by Legate.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${Legion_CUDA_ARCH}")
endif()

if(Legion_USE_CUDA)
  include(Modules/cuda_arch_helpers)
  # Needs to run before `rapids_cuda_init_architectures`
  set_cuda_arch_from_names()
  # Needs to run before `enable_language(CUDA)`
  rapids_cuda_init_architectures(cupynumeric)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
  enable_language(CUDA)
  # Since cupynumeric only enables CUDA optionally we need to manually include
  # the file that rapids_cuda_init_architectures relies on `project` calling
  if(CMAKE_PROJECT_cupynumeric_INCLUDE)
    include("${CMAKE_PROJECT_cupynumeric_INCLUDE}")
  endif()

  # Must come after enable_language(CUDA)
  # Use `-isystem <path>` instead of `-isystem=<path>`
  # because the former works with clangd intellisense
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET cupynumeric-exports
    INSTALL_EXPORT_SET cupynumeric-exports
  )

  include(thirdparty/get_nccl)
  include(thirdparty/get_cutensor)
endif()

include(thirdparty/get_openblas)

include(thirdparty/get_tblis)

##############################################################################
# - cuPyNumeric ----------------------------------------------------------------

add_library(cupynumeric)
add_library(cupynumeric::cupynumeric ALIAS cupynumeric)

set(cupynumeric_CXX_OPTIONS "")
set(cupynumeric_CUDA_OPTIONS "")

include(Modules/set_cpu_arch_flags)
set_cpu_arch_flags(cupynumeric_CXX_OPTIONS)

# Add `src/cupynumeric.mk` sources
target_sources(cupynumeric PRIVATE
  src/cupynumeric/ternary/where.cc
  src/cupynumeric/scan/scan_global.cc
  src/cupynumeric/scan/scan_local.cc
  src/cupynumeric/binary/binary_op.cc
  src/cupynumeric/binary/binary_op_util.cc
  src/cupynumeric/binary/binary_red.cc
  src/cupynumeric/bits/packbits.cc
  src/cupynumeric/bits/unpackbits.cc
  src/cupynumeric/unary/scalar_unary_red.cc
  src/cupynumeric/unary/unary_op.cc
  src/cupynumeric/unary/unary_red.cc
  src/cupynumeric/unary/convert.cc
  src/cupynumeric/nullary/arange.cc
  src/cupynumeric/nullary/eye.cc
  src/cupynumeric/nullary/fill.cc
  src/cupynumeric/nullary/window.cc
  src/cupynumeric/index/advanced_indexing.cc
  src/cupynumeric/index/choose.cc
  src/cupynumeric/index/putmask.cc
  src/cupynumeric/index/repeat.cc
  src/cupynumeric/index/select.cc
  src/cupynumeric/index/wrap.cc
  src/cupynumeric/index/zip.cc
  src/cupynumeric/item/read.cc
  src/cupynumeric/item/write.cc
  src/cupynumeric/matrix/batched_cholesky.cc
  src/cupynumeric/matrix/contract.cc
  src/cupynumeric/matrix/diag.cc
  src/cupynumeric/matrix/geev.cc
  src/cupynumeric/matrix/gemm.cc
  src/cupynumeric/matrix/matmul.cc
  src/cupynumeric/matrix/matvecmul.cc
  src/cupynumeric/matrix/dot.cc
  src/cupynumeric/matrix/potrf.cc
  src/cupynumeric/matrix/qr.cc
  src/cupynumeric/matrix/solve.cc
  src/cupynumeric/matrix/svd.cc
  src/cupynumeric/matrix/syrk.cc
  src/cupynumeric/matrix/tile.cc
  src/cupynumeric/matrix/transpose.cc
  src/cupynumeric/matrix/trilu.cc
  src/cupynumeric/matrix/trsm.cc
  src/cupynumeric/matrix/util.cc
  src/cupynumeric/random/bitgenerator.cc
  src/cupynumeric/random/randutil/generator_host.cc
  src/cupynumeric/random/randutil/generator_host_straightforward.cc
  src/cupynumeric/random/randutil/generator_host_advanced.cc
  src/cupynumeric/random/rand.cc
  src/cupynumeric/search/argwhere.cc
  src/cupynumeric/search/nonzero.cc
  src/cupynumeric/set/unique.cc
  src/cupynumeric/set/unique_reduce.cc
  src/cupynumeric/stat/bincount.cc
  src/cupynumeric/convolution/convolve.cc
  src/cupynumeric/transform/flip.cc
  src/cupynumeric/utilities/repartition.cc
  src/cupynumeric/arg_redop_register.cc
  src/cupynumeric/mapper.cc
  src/cupynumeric/ndarray.cc
  src/cupynumeric/operators.cc
  src/cupynumeric/runtime.cc
  src/cupynumeric/cephes/chbevl.cc
  src/cupynumeric/cephes/i0.cc
  src/cupynumeric/stat/histogram.cc
)

if(Legion_USE_OpenMP)
  target_sources(cupynumeric PRIVATE
    src/cupynumeric/ternary/where_omp.cc
    src/cupynumeric/scan/scan_global_omp.cc
    src/cupynumeric/scan/scan_local_omp.cc
    src/cupynumeric/binary/binary_op_omp.cc
    src/cupynumeric/binary/binary_red_omp.cc
    src/cupynumeric/bits/packbits_omp.cc
    src/cupynumeric/bits/unpackbits_omp.cc
    src/cupynumeric/unary/unary_op_omp.cc
    src/cupynumeric/unary/scalar_unary_red_omp.cc
    src/cupynumeric/unary/unary_red_omp.cc
    src/cupynumeric/unary/convert_omp.cc
    src/cupynumeric/nullary/arange_omp.cc
    src/cupynumeric/nullary/eye_omp.cc
    src/cupynumeric/nullary/fill_omp.cc
    src/cupynumeric/nullary/window_omp.cc
    src/cupynumeric/index/advanced_indexing_omp.cc
    src/cupynumeric/index/choose_omp.cc
    src/cupynumeric/index/putmask_omp.cc
    src/cupynumeric/index/repeat_omp.cc
    src/cupynumeric/index/select_omp.cc
    src/cupynumeric/index/wrap_omp.cc
    src/cupynumeric/index/zip_omp.cc
    src/cupynumeric/matrix/batched_cholesky_omp.cc
    src/cupynumeric/matrix/contract_omp.cc
    src/cupynumeric/matrix/diag_omp.cc
    src/cupynumeric/matrix/geev_omp.cc
    src/cupynumeric/matrix/gemm_omp.cc
    src/cupynumeric/matrix/matmul_omp.cc
    src/cupynumeric/matrix/matvecmul_omp.cc
    src/cupynumeric/matrix/dot_omp.cc
    src/cupynumeric/matrix/potrf_omp.cc
    src/cupynumeric/matrix/qr_omp.cc
    src/cupynumeric/matrix/solve_omp.cc
    src/cupynumeric/matrix/svd_omp.cc
    src/cupynumeric/matrix/syrk_omp.cc
    src/cupynumeric/matrix/tile_omp.cc
    src/cupynumeric/matrix/transpose_omp.cc
    src/cupynumeric/matrix/trilu_omp.cc
    src/cupynumeric/matrix/trsm_omp.cc
    src/cupynumeric/random/rand_omp.cc
    src/cupynumeric/search/argwhere_omp.cc
    src/cupynumeric/search/nonzero_omp.cc
    src/cupynumeric/set/unique_omp.cc
    src/cupynumeric/set/unique_reduce_omp.cc
    src/cupynumeric/stat/bincount_omp.cc
    src/cupynumeric/convolution/convolve_omp.cc
    src/cupynumeric/transform/flip_omp.cc
    src/cupynumeric/stat/histogram_omp.cc
  )
endif()

if(Legion_USE_CUDA)
  target_sources(cupynumeric PRIVATE
    src/cupynumeric/ternary/where.cu
    src/cupynumeric/scan/scan_global.cu
    src/cupynumeric/scan/scan_local.cu
    src/cupynumeric/binary/binary_op.cu
    src/cupynumeric/binary/binary_red.cu
    src/cupynumeric/bits/packbits.cu
    src/cupynumeric/bits/unpackbits.cu
    src/cupynumeric/unary/scalar_unary_red.cu
    src/cupynumeric/unary/unary_red.cu
    src/cupynumeric/unary/unary_op.cu
    src/cupynumeric/unary/convert.cu
    src/cupynumeric/nullary/arange.cu
    src/cupynumeric/nullary/eye.cu
    src/cupynumeric/nullary/fill.cu
    src/cupynumeric/nullary/window.cu
    src/cupynumeric/index/advanced_indexing.cu
    src/cupynumeric/index/choose.cu
    src/cupynumeric/index/putmask.cu
    src/cupynumeric/index/repeat.cu
    src/cupynumeric/index/select.cu
    src/cupynumeric/index/wrap.cu
    src/cupynumeric/index/zip.cu
    src/cupynumeric/item/read.cu
    src/cupynumeric/item/write.cu
    src/cupynumeric/matrix/batched_cholesky.cu
    src/cupynumeric/matrix/contract.cu
    src/cupynumeric/matrix/diag.cu
    src/cupynumeric/matrix/geev.cu
    src/cupynumeric/matrix/gemm.cu
    src/cupynumeric/matrix/matmul.cu
    src/cupynumeric/matrix/matvecmul.cu
    src/cupynumeric/matrix/dot.cu
    src/cupynumeric/matrix/potrf.cu
    src/cupynumeric/matrix/qr.cu
    src/cupynumeric/matrix/solve.cu
    src/cupynumeric/matrix/svd.cu
    src/cupynumeric/matrix/syrk.cu
    src/cupynumeric/matrix/tile.cu
    src/cupynumeric/matrix/transpose.cu
    src/cupynumeric/matrix/trilu.cu
    src/cupynumeric/matrix/trsm.cu
    src/cupynumeric/random/rand.cu
    src/cupynumeric/search/argwhere.cu
    src/cupynumeric/search/nonzero.cu
    src/cupynumeric/set/unique.cu
    src/cupynumeric/stat/bincount.cu
    src/cupynumeric/convolution/convolve.cu
    src/cupynumeric/fft/fft.cu
    src/cupynumeric/transform/flip.cu
    src/cupynumeric/utilities/repartition.cu
    src/cupynumeric/arg_redop_register.cu
    src/cupynumeric/cudalibs.cu
    src/cupynumeric/stat/histogram.cu
  )
endif()

# Add `src/cupynumeric/sort/sort.mk` sources
target_sources(cupynumeric PRIVATE
  src/cupynumeric/sort/sort.cc
  src/cupynumeric/sort/searchsorted.cc
)

if(Legion_USE_OpenMP)
  target_sources(cupynumeric PRIVATE
    src/cupynumeric/sort/sort_omp.cc
    src/cupynumeric/sort/searchsorted_omp.cc
  )
endif()

if(Legion_USE_CUDA)
  target_sources(cupynumeric PRIVATE
    src/cupynumeric/sort/sort.cu
    src/cupynumeric/sort/searchsorted.cu
    src/cupynumeric/sort/cub_sort_bool.cu
    src/cupynumeric/sort/cub_sort_int8.cu
    src/cupynumeric/sort/cub_sort_int16.cu
    src/cupynumeric/sort/cub_sort_int32.cu
    src/cupynumeric/sort/cub_sort_int64.cu
    src/cupynumeric/sort/cub_sort_uint8.cu
    src/cupynumeric/sort/cub_sort_uint16.cu
    src/cupynumeric/sort/cub_sort_uint32.cu
    src/cupynumeric/sort/cub_sort_uint64.cu
    src/cupynumeric/sort/cub_sort_half.cu
    src/cupynumeric/sort/cub_sort_float.cu
    src/cupynumeric/sort/cub_sort_double.cu
    src/cupynumeric/sort/thrust_sort_bool.cu
    src/cupynumeric/sort/thrust_sort_int8.cu
    src/cupynumeric/sort/thrust_sort_int16.cu
    src/cupynumeric/sort/thrust_sort_int32.cu
    src/cupynumeric/sort/thrust_sort_int64.cu
    src/cupynumeric/sort/thrust_sort_uint8.cu
    src/cupynumeric/sort/thrust_sort_uint16.cu
    src/cupynumeric/sort/thrust_sort_uint32.cu
    src/cupynumeric/sort/thrust_sort_uint64.cu
    src/cupynumeric/sort/thrust_sort_half.cu
    src/cupynumeric/sort/thrust_sort_float.cu
    src/cupynumeric/sort/thrust_sort_double.cu
    src/cupynumeric/sort/thrust_sort_complex64.cu
    src/cupynumeric/sort/thrust_sort_complex128.cu
  )
endif()

# Add `src/cupynumeric/random/random.mk` sources
if(Legion_USE_CUDA)
  target_sources(cupynumeric PRIVATE
      src/cupynumeric/random/bitgenerator.cu
      src/cupynumeric/random/randutil/generator_device.cu
      src/cupynumeric/random/randutil/generator_device_straightforward.cu
      src/cupynumeric/random/randutil/generator_device_advanced.cu
  )
endif()

# add sources for cusolverMp
if(Legion_USE_CUDA AND CUSOLVERMP_DIR)
  target_sources(cupynumeric PRIVATE
    src/cupynumeric/matrix/mp_potrf.cu
    src/cupynumeric/matrix/mp_solve.cu
  )
endif()

target_sources(cupynumeric PRIVATE
  # This must always be the last file!
  # It guarantees we do our registration callback
  # only after all task variants are recorded
  src/cupynumeric/cupynumeric.cc
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(cupynumeric PUBLIC  "$<$<COMPILE_LANGUAGE:CXX,CUDA>:DEBUG_CUPYNUMERIC>")
endif()

if(Legion_BOUNDS_CHECKS)
  target_compile_definitions(cupynumeric PUBLIC "$<$<COMPILE_LANGUAGE:CXX,CUDA>:BOUNDS_CHECKS>")
endif()

list(APPEND cupynumeric_CUDA_OPTIONS -Xfatbin=-compress-all)
list(APPEND cupynumeric_CUDA_OPTIONS --expt-extended-lambda)
list(APPEND cupynumeric_CUDA_OPTIONS --expt-relaxed-constexpr)
list(APPEND cupynumeric_CXX_OPTIONS -Wno-deprecated-declarations)
list(APPEND cupynumeric_CUDA_OPTIONS -Wno-deprecated-declarations)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(platform_rpath_origin "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(platform_rpath_origin "@loader_path")
endif ()

set_target_properties(cupynumeric
           PROPERTIES BUILD_RPATH                         "${platform_rpath_origin}"
                      INSTALL_RPATH                       "${platform_rpath_origin}"
                      CXX_STANDARD                        17
                      CXX_STANDARD_REQUIRED               ON
                      POSITION_INDEPENDENT_CODE           ON
                      INTERFACE_POSITION_INDEPENDENT_CODE ON
                      CUDA_STANDARD                       17
                      CUDA_STANDARD_REQUIRED              ON
                      LIBRARY_OUTPUT_DIRECTORY            lib)

target_link_libraries(cupynumeric
  PUBLIC legate::legate
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

if(NOT Legion_USE_CUDA AND cupynumeric_cuRAND_INCLUDE_DIR)
  target_compile_definitions(cupynumeric
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:CUPYNUMERIC_CURAND_FOR_CPU_BUILD>")
  target_include_directories(cupynumeric PRIVATE ${cupynumeric_cuRAND_INCLUDE_DIR})
endif()

if(Legion_USE_CUDA AND CUSOLVERMP_DIR)
  message(VERBOSE "cupynumeric: CUSOLVERMP_DIR ${CUSOLVERMP_DIR}")
  target_compile_definitions(cupynumeric PUBLIC "$<$<COMPILE_LANGUAGE:CXX,CUDA>:CUPYNUMERIC_USE_CUSOLVERMP>")
  target_include_directories(cupynumeric PRIVATE ${CUSOLVERMP_DIR}/include)
  target_link_libraries(cupynumeric PRIVATE ${CUSOLVERMP_DIR}/lib/libcusolverMp.so)
endif()

target_compile_options(cupynumeric
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${cupynumeric_CXX_OPTIONS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${cupynumeric_CUDA_OPTIONS}>")

target_include_directories(cupynumeric
  PUBLIC
    $<BUILD_INTERFACE:${cupynumeric_SOURCE_DIR}/src>
  INTERFACE
    $<INSTALL_INTERFACE:include/cupynumeric>
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
  target_link_options(cupynumeric PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
endif()

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS cupynumeric
        DESTINATION ${lib_dir}
        EXPORT cupynumeric-exports)

install(
  FILES src/cupynumeric.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/cupynumeric/version_config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cupynumeric)

install(
  FILES src/cupynumeric/cupynumeric_c.h
        src/cupynumeric/ndarray.h
        src/cupynumeric/ndarray.inl
        src/cupynumeric/operators.h
        src/cupynumeric/operators.inl
        src/cupynumeric/runtime.h
        src/cupynumeric/slice.h
        src/cupynumeric/typedefs.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cupynumeric/cupynumeric)

if(cupynumeric_INSTALL_TBLIS)
  install(DIRECTORY ${tblis_BINARY_DIR}/lib/ DESTINATION ${lib_dir})
  install(DIRECTORY ${tblis_BINARY_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide targets for cuPyNumeric, an aspiring drop-in replacement for NumPy at scale.

Imported Targets:
  - cupynumeric::cupynumeric

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
  INSTALL cupynumeric
  EXPORT_SET cupynumeric-exports
  GLOBAL_TARGETS cupynumeric
  NAMESPACE cupynumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD cupynumeric
  EXPORT_SET cupynumeric-exports
  GLOBAL_TARGETS cupynumeric
  NAMESPACE cupynumeric::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

if(cupynumeric_BUILD_TESTS)
  include(CTest)

  add_subdirectory(tests/cpp)
endif()
