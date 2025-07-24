#!/usr/bin/env python

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
#
from __future__ import annotations

import argparse
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Flush output on newlines
sys.stdout.reconfigure(line_buffering=True)

os_name = platform.system()

if os_name == "Linux":
    pass
elif os_name == "Darwin":
    pass
else:
    raise Exception("install.py script does not work on %s" % os_name)


class BooleanFlag(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        default,
        required=False,
        help="",
        metavar=None,
    ):
        assert all(not opt.startswith("--no") for opt in option_strings)

        def flatten(list):
            return [item for sublist in list for item in sublist]

        option_strings = flatten(
            [
                (
                    [opt, "--no-" + opt[2:], "--no" + opt[2:]]
                    if opt.startswith("--")
                    else [opt]
                )
                for opt in option_strings
            ]
        )
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            const=None,
            default=default,
            type=bool,
            choices=None,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, not option_string.startswith("--no"))


def execute_command(args, verbose, ignore_errors=False, **kwargs):
    if verbose:
        print('Executing: "', " ".join(args), '" with ', kwargs)
    if ignore_errors:
        subprocess.call(args, **kwargs)
    else:
        subprocess.check_call(args, **kwargs)


def scikit_build_cmake_build_dir(skbuild_dir):
    if os.path.exists(skbuild_dir):
        for f in os.listdir(skbuild_dir):
            if os.path.exists(
                cmake_build := os.path.join(skbuild_dir, f, "cmake-build")
            ):
                return cmake_build
    return None


def find_cmake_val(pattern, filepath):
    return (
        subprocess.check_output(["grep", "--color=never", pattern, filepath])
        .decode("UTF-8")
        .strip()
    )


def was_previously_built_with_different_build_isolation(
    isolated, cupynumeric_build_dir
):
    if (
        cupynumeric_build_dir is not None
        and os.path.exists(cupynumeric_build_dir)
        and os.path.exists(
            cmake_cache := os.path.join(
                cupynumeric_build_dir, "CMakeCache.txt"
            )
        )
    ):
        try:
            if isolated:
                return True
            if find_cmake_val("pip-build-env", cmake_cache):
                return True
        except Exception:
            pass
    return False


def find_legate_cmake_dir() -> Path:
    r"""Try to determine the location of legate cmake files.

    Returns
    -------
    Path
        The directory containing the legate cmake files.

    Raises
    ------
    RuntimeError
        If legate cmake directory could not be found.
    """
    try:
        import legate.install_info as lg_install_info
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "Cannot determine Legate install directory. Please make sure "
            "Legate is installed in the current Python environment."
        ) from e

    path = Path(lg_install_info.libpath).resolve()
    if (path / "cmake" / "legate").exists():
        # If this exists, then we were installed normally into a python or
        # conda env.
        return path

    # Possibly installed in an editable installation, in which case legate
    # config.cmake and friends will live in the root binary directory.
    root_path = path.root
    assert isinstance(root_path, str)
    while not any(p.name == "legate-config.cmake" for p in path.iterdir()):
        path = path.parent
        if str(path) == root_path:
            raise RuntimeError(
                "Could not determine directory containing legate CMake files"
            )
    return path


def install_cupynumeric(
    arch,
    build_isolation,
    with_tests,
    check_bounds,
    clean_first,
    cmake_exe,
    cmake_generator,
    conduit,
    cuda_dir,
    cuda,
    curand_dir,
    cusolvermp_dir,
    cutensor_dir,
    debug_release,
    debug,
    editable,
    extra_flags,
    gasnet_dir,
    networks,
    hdf,
    llvm,
    march,
    maxdim,
    maxfields,
    nccl_dir,
    openblas_dir,
    openmp,
    spy,
    tblis_dir,
    thread_count,
    unknown,
    verbose,
):
    if len(networks) > 1:
        print(
            "Warning: Building Realm with multiple networking backends is not "
            "fully supported currently."
        )

    if clean_first is None:
        clean_first = not editable

    print("Verbose build is ", "on" if verbose else "off")
    if verbose:
        print("Options are:")
        print("arch: ", arch)
        print("build_isolation: ", build_isolation)
        print("with_tests: ", with_tests)
        print("check_bounds: ", check_bounds)
        print("clean_first: ", clean_first)
        print("cmake_exe: ", cmake_exe)
        print("cmake_generator: ", cmake_generator)
        print("conduit: ", conduit)
        print("cuda_dir: ", cuda_dir)
        print("cuda: ", cuda)
        print("curand_dir: ", curand_dir)
        print("cusolvermp_dir: ", cusolvermp_dir)
        print("cutensor_dir: ", cutensor_dir)
        print("debug_release: ", debug_release)
        print("debug: ", debug)
        print("editable: ", editable)
        print("extra_flags: ", extra_flags)
        print("gasnet_dir: ", gasnet_dir)
        print("networks: ", networks)
        print("hdf: ", hdf)
        print("llvm: ", llvm)
        print("march: ", march)
        print("maxdim: ", maxdim)
        print("maxfields: ", maxfields)
        print("nccl_dir: ", nccl_dir)
        print("openblas_dir: ", openblas_dir)
        print("openmp: ", openmp)
        print("spy: ", spy)
        print("tblis_dir: ", tblis_dir)
        print("thread_count: ", thread_count)
        print("unknown: ", unknown)
        print("verbose: ", verbose)

    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    realpath = os.path.realpath

    cupynumeric_dir = dirname(realpath(__file__))

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    def validate_path(path):
        if path is None or (path := str(path)) == "":
            return None
        if not os.path.isabs(path):
            path = join(cupynumeric_dir, path)
        if not exists(path := realpath(path)):
            print(f"Error: path does not exist: {path}")
            sys.exit(1)
        return path

    cuda_dir = validate_path(cuda_dir)
    nccl_dir = validate_path(nccl_dir)
    tblis_dir = validate_path(tblis_dir)
    curand_dir = validate_path(curand_dir)
    gasnet_dir = validate_path(gasnet_dir)
    cusolvermp_dir = validate_path(cusolvermp_dir)
    cutensor_dir = validate_path(cutensor_dir)
    openblas_dir = validate_path(openblas_dir)

    legate_dir = find_legate_cmake_dir()

    if verbose:
        print("cuda_dir: ", cuda_dir)
        print("nccl_dir: ", nccl_dir)
        print("tblis_dir: ", tblis_dir)
        print("legate_dir: ", legate_dir)
        print("curand_dir: ", curand_dir)
        print("gasnet_dir: ", gasnet_dir)
        print("cusolvermp_dir: ", cusolvermp_dir)
        print("cutensor_dir: ", cutensor_dir)
        print("openblas_dir: ", openblas_dir)

    skbuild_dir = join(cupynumeric_dir, "_skbuild")
    cupynumeric_build_dir = scikit_build_cmake_build_dir(skbuild_dir)

    if was_previously_built_with_different_build_isolation(
        build_isolation and not editable, cupynumeric_build_dir
    ):
        print("Performing a clean build to accommodate build isolation.")
        clean_first = True

    cmd_env = dict(os.environ.items())

    # Explicitly uninstall cupynumeric if doing a clean/isolated build.
    #
    # A prior installation may have built and installed cupynumeric C++
    # dependencies (like BLAS or tblis).
    #
    # CMake will find and use them for the current build, which would normally
    # be correct, but pip uninstalls files from any existing installation as
    # the last step of the install process, including the libraries found by
    # CMake during the current build.
    #
    # Therefore this uninstall step must occur *before* CMake attempts to find
    # these dependencies, triggering CMake to build and install them again.
    if clean_first or (build_isolation and not editable):
        execute_command(
            [sys.executable, "-m", "pip", "uninstall", "-y", "cupynumeric"],
            verbose,
            ignore_errors=True,
            cwd=cupynumeric_dir,
            env=cmd_env,
        )

    if clean_first:
        shutil.rmtree(skbuild_dir, ignore_errors=True)
        shutil.rmtree(join(cupynumeric_dir, "dist"), ignore_errors=True)
        shutil.rmtree(join(cupynumeric_dir, "build"), ignore_errors=True)
        shutil.rmtree(
            join(cupynumeric_dir, "cupynumeric.egg-info"), ignore_errors=True
        )

    # Configure and build cuPyNumeric via setup.py
    pip_install_cmd = [sys.executable, "-m", "pip", "install"]

    install_dir = None

    if unknown is not None:
        try:
            prefix_loc = unknown.index("--prefix")
            prefix_dir = validate_path(unknown[prefix_loc + 1])
            if prefix_dir is not None:
                install_dir = prefix_dir
                unknown = unknown[:prefix_loc] + unknown[prefix_loc + 2 :]
        except Exception:
            pass

    install_dir = validate_path(install_dir)

    if verbose:
        print("install_dir: ", install_dir)

    if install_dir is not None:
        pip_install_cmd += ["--root", "/", "--prefix", str(install_dir)]

    if editable:
        # editable implies build_isolation = False
        pip_install_cmd += ["--no-deps", "--no-build-isolation", "--editable"]
        cmd_env.update({"SETUPTOOLS_ENABLE_FEATURES": "legacy-editable"})
    else:
        if not build_isolation:
            pip_install_cmd += ["--no-deps", "--no-build-isolation"]
        pip_install_cmd += ["--upgrade"]

    if unknown is not None:
        pip_install_cmd += unknown

    pip_install_cmd += ["."]
    if verbose:
        pip_install_cmd += ["-vv"]

    # Also use preexisting CMAKE_ARGS from conda if set
    cmake_flags = cmd_env.get("CMAKE_ARGS", "").split(" ")

    if debug or verbose:
        cmake_flags += ["--log-level=%s" % ("DEBUG" if debug else "VERBOSE")]

    cmake_flags += f"""\
-DCMAKE_BUILD_TYPE={
        (
            "Debug"
            if debug
            else "RelWithDebInfo"
            if debug_release
            else "Release"
        )
    }
-DBUILD_SHARED_LIBS=ON
-DCMAKE_CUDA_ARCHITECTURES={str(arch)}
-DLegion_MAX_DIM={str(maxdim)}
-DLegion_MAX_FIELDS={str(maxfields)}
-DLegion_SPY={("ON" if spy else "OFF")}
-DLegion_BOUNDS_CHECKS={("ON" if check_bounds else "OFF")}
-DLegion_USE_CUDA={("ON" if cuda else "OFF")}
-DLegion_USE_OpenMP={("ON" if openmp else "OFF")}
-DLegion_USE_LLVM={("ON" if llvm else "OFF")}
-DLegion_NETWORKS={";".join(networks)}
-DLegion_USE_HDF5={("ON" if hdf else "OFF")}
-Dcupynumeric_BUILD_TESTS={("ON" if with_tests else "OFF")}
""".splitlines()

    if march:
        cmake_flags += [f"-DBUILD_MARCH={march}"]
    if cuda_dir:
        cmake_flags += ["-DCUDAToolkit_ROOT=%s" % cuda_dir]
    if nccl_dir:
        cmake_flags += ["-DNCCL_DIR=%s" % nccl_dir]
    if gasnet_dir:
        cmake_flags += ["-DGASNet_ROOT_DIR=%s" % gasnet_dir]
    if conduit:
        cmake_flags += ["-DGASNet_CONDUIT=%s" % conduit]
    if tblis_dir:
        cmake_flags += ["-Dtblis_ROOT=%s" % tblis_dir]
    if openblas_dir:
        cmake_flags += ["-DBLAS_DIR=%s" % openblas_dir]
    if cusolvermp_dir:
        cmake_flags += ["-DCUSOLVERMP_DIR=%s" % cusolvermp_dir]
    if cutensor_dir:
        cmake_flags += ["-Dcutensor_DIR=%s" % cutensor_dir]
    # A custom path to cuRAND is ignored when CUDA support is available
    if cuda and curand_dir is not None:
        cmake_flags += ["-Dcupynumeric_cuRAND_INCLUDE_DIR=%s" % curand_dir]

    cmake_flags += ["-Dlegate_ROOT=%s" % str(legate_dir)]
    cmake_flags += ["-DCMAKE_BUILD_PARALLEL_LEVEL=%s" % thread_count]

    cmake_flags += extra_flags
    build_flags = [f"-j{str(thread_count)}"]
    if verbose:
        if cmake_generator == "Unix Makefiles":
            build_flags += ["VERBOSE=1"]
        else:
            build_flags += ["--verbose"]

    cmd_env.update(
        {
            "CMAKE_ARGS": " ".join(cmake_flags),
            "CMAKE_GENERATOR": cmake_generator,
            "SKBUILD_BUILD_OPTIONS": " ".join(build_flags),
        }
    )

    execute_command(pip_install_cmd, verbose, cwd=cupynumeric_dir, env=cmd_env)


def driver():
    parser = argparse.ArgumentParser(description="Install cuPyNumeric.")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG", "0") == "1",
        help="Build cuPyNumeric with no optimizations.",
    )
    parser.add_argument(
        "--debug-release",
        dest="debug_release",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG_RELEASE", "0") == "1",
        help="Build cuPyNumeric with optimizations, but include debugging "
        "symbols.",
    )
    parser.add_argument(
        "--with-tests",
        dest="with_tests",
        action="store_true",
        required=False,
        default=False,
        help="Build cuPyNumeric tests.",
    )
    parser.add_argument(
        "--check-bounds",
        dest="check_bounds",
        action="store_true",
        required=False,
        default=False,
        help="Build cuPyNumeric with bounds checks.",
    )
    parser.add_argument(
        "--max-dim",
        dest="maxdim",
        type=int,
        default=int(os.environ.get("LEGION_MAX_DIM", 4)),
        help="Maximum number of dimensions that cuPyNumeric will support",
    )
    parser.add_argument(
        "--max-fields",
        dest="maxfields",
        type=int,
        default=int(os.environ.get("LEGION_MAX_FIELDS", 256)),
        help="Maximum number of fields that cuPyNumeric will support",
    )
    parser.add_argument(
        "--network",
        dest="networks",
        action="append",
        required=False,
        choices=["gasnet1", "gasnetex", "mpi"],
        default=[],
        help="Realm networking backend to use for multi-node execution.",
    )
    parser.add_argument(
        "--with-gasnet",
        dest="gasnet_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("GASNET"),
        help="Path to GASNet installation directory.",
    )
    parser.add_argument(
        "--with-openblas",
        dest="openblas_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("OPENBLAS_PATH"),
        help="Path to OpenBLAS installation directory. Note that providing a "
        "user-defined BLAS library may lead to dynamic library conflicts with "
        "BLAS loaded by Python's Numpy. When using cuPyNumeric's BLAS, this "
        "issue is prevented by a custom library name.",
    )
    parser.add_argument(
        "--with-tblis",
        dest="tblis_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("TBLIS_PATH"),
        help="Path to TBLIS installation directory.",
    )
    parser.add_argument(
        "--with-curand",
        dest="curand_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("CURAND_PATH"),
        help="Path to cuRAND installation directory. This flag is ignored "
        "if Legate was built with CUDA support.",
    )
    # TODO(jfaibussowit) maybe split to --with-cusolvermp [bool]
    # and a --with-cusolvermp-dir [dir]
    parser.add_argument(
        "--with-cusolvermp",
        dest="cusolvermp_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("CUSOLVERMP_PATH"),
        help="Path to cuSolverMp installation directory.",
    )
    parser.add_argument(
        "--with-cutensor",
        dest="cutensor_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("CUTENSOR_PATH"),
        help="Path to cuTensor installation directory.",
    )
    parser.add_argument(
        "--with-nccl",
        dest="nccl_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("NCCL_PATH"),
        help="Path to NCCL installation directory.",
    )
    parser.add_argument(
        "--with-cmake",
        dest="cmake_exe",
        metavar="EXE",
        required=False,
        default="cmake",
        help="Path to CMake executable (if not on PATH).",
    )
    parser.add_argument(
        "--cmake-generator",
        dest="cmake_generator",
        required=False,
        default=os.environ.get(
            "CMAKE_GENERATOR",
            "Unix Makefiles" if shutil.which("ninja") is None else "Ninja",
        ),
        choices=["Ninja", "Unix Makefiles", None],
        help="The CMake makefiles generator",
    )
    parser.add_argument(
        "--cuda",
        action=BooleanFlag,
        default=os.environ.get("USE_CUDA", "0") == "1",
        help="Build cuPyNumeric with CUDA support.",
    )
    parser.add_argument(
        "--with-cuda",
        dest="cuda_dir",
        metavar="DIR",
        required=False,
        default=os.environ.get("CUDA"),
        help="Path to CUDA installation directory.",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        action="store",
        required=False,
        default="all-major",
        help="Specify the target GPU architecture.",
    )
    parser.add_argument(
        "--openmp",
        action=BooleanFlag,
        default=os.environ.get("USE_OPENMP", "0") == "1",
        help="Build cuPyNumeric with OpenMP support.",
    )
    parser.add_argument(
        "--march",
        dest="march",
        required=False,
        default=("haswell" if platform.machine() == "x86_64" else None),
        help="Specify the target CPU architecture.",
    )
    parser.add_argument(
        "--llvm",
        dest="llvm",
        action="store_true",
        required=False,
        default=os.environ.get("USE_LLVM", "0") == "1",
        help="Build cuPyNumeric with LLVM support.",
    )
    parser.add_argument(
        "--hdf5",
        "--hdf",
        dest="hdf",
        action="store_true",
        required=False,
        default=os.environ.get("USE_HDF", "0") == "1",
        help="Build cuPyNumeric with HDF support.",
    )
    parser.add_argument(
        "--spy",
        dest="spy",
        action="store_true",
        required=False,
        default=os.environ.get("USE_SPY", "0") == "1",
        help="Build cuPyNumeric with detailed Legion Spy enabled.",
    )
    parser.add_argument(
        "--conduit",
        dest="conduit",
        action="store",
        required=False,
        # TODO: To support UDP conduit, we would need to add a special case on
        # the legate launcher.
        # See https://github.com/nv-legate/legate.core/issues/294.
        choices=["ibv", "ucx", "aries", "mpi"],
        default=os.environ.get("CONDUIT"),
        help="Build cuPyNumeric with specified GASNet conduit.",
    )
    parser.add_argument(
        "--clean",
        dest="clean_first",
        action=BooleanFlag,
        default=None,
        help="Clean before build.",
    )
    parser.add_argument(
        "--extra",
        dest="extra_flags",
        action="append",
        required=False,
        default=[],
        help="Extra CMake flags.",
    )
    parser.add_argument(
        "-j",
        dest="thread_count",
        nargs="?",
        type=int,
        required=False,
        default=os.environ.get("CPU_COUNT"),
        help="Number of threads used to compile.",
    )
    parser.add_argument(
        "--editable",
        dest="editable",
        action="store_true",
        required=False,
        default=False,
        help="Perform an editable install. Disables --build-isolation if set "
        "(passing --no-deps --no-build-isolation to pip).",
    )
    parser.add_argument(
        "--build-isolation",
        dest="build_isolation",
        action=BooleanFlag,
        required=False,
        default=True,
        help="Enable isolation when building a modern source distribution. "
        "Build dependencies specified by PEP 518 must be already "
        "installed if this option is used.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose build output.",
    )
    args, unknown = parser.parse_known_args()

    install_cupynumeric(unknown=unknown, **vars(args))


if __name__ == "__main__":
    driver()
