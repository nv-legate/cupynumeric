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


"""
Test that verifies unimplemented functions are properly tracked in Sphinx config.

After removing NumPy fallback, unimplemented functions should:
1. Still appear in the Sphinx config (for documentation completeness)
2. Not exist in cuPyNumeric's namespace (will show "-" in docs table)

This test ensures that:
1. All functions marked as NOT IMPLEMENTED are truly missing from cuPyNumeric
2. All functions marked as NOT IMPLEMENTED are still in the Sphinx config
3. All other functions in the config exist in cuPyNumeric
"""

import pytest

import cupynumeric as cn
from cupynumeric._sphinxext._comparison_config import (
    CONVOLVE,
    CREATION,
    CREATION_ND,
    EINSUM,
    FACTOR,
    FUNCTIONAL,
    INDEX,
    IO,
    IO_ND,
    LOGICAL,
    LU,
    MANIP,
    MANIP_ND,
    MATH,
    MISC,
    PAD,
    PACK,
    SEARCHING,
    SET,
    STATS,
    SVD,
)


# Functions that SHOULD BE unimplemented (in config but not in cuPyNumeric)
# These appear in the Sphinx docs table but show "-" in the cuPyNumeric column
EXPECTED_UNIMPLEMENTED = {
    # CONVOLVE
    "correlate",
    # LOGICAL
    "array_equiv",
    "isfortran",
    # SET
    "intersect1d",
    "setdiff1d",
    "setxor1d",
    "union1d",
    # MANIP
    "asarray_chkfinite",
    "asmatrix",
    "require",
    "resize",
    "rollaxis",
    "trim_zeros",
    # SVD
    "lstsq",
    "matrix_rank",
    # LU
    "det",
    "inv",
    "slogdet",
    "tensorinv",
    "tensorsolve",
    # CREATION
    "bmat",
    "fromfunction",
    "geomspace",
    "vander",
    # IO
    "array_repr",
    "array_str",
    "array2string",
    "base_repr",
    "binary_repr",
    "format_float_positional",
    "format_float_scientific",
    "fromregex",
    "genfromtxt",
    "get_printoptions",
    "loadtxt",
    "printoptions",
    "save",
    "savetxt",
    "savez_compressed",
    "savez",
    "set_printoptions",
    # MATH
    "around",
    "ediff1d",
    "fix",
    "i0",
    "interp",
    "sinc",
    "trapz",
    "unwrap",
    # STATS
    "corrcoef",
    "histogram_bin_edges",
    "nanstd",
    "nanvar",
    "ptp",
    "std",
    # MISC
    "kron",
    # FUNCTIONAL
    "apply_along_axis",
    "apply_over_axes",
    "piecewise",
}

# All function categories with their namespaces
ALL_CATEGORIES = {
    "CONVOLVE": (CONVOLVE, None),
    "LOGICAL": (LOGICAL, None),
    "EINSUM": (EINSUM, None),
    "SET": (SET, None),
    "MANIP": (MANIP, None),
    "MANIP_ND": (MANIP_ND, "ndarray"),
    "FACTOR": (FACTOR, "linalg"),
    "SVD": (SVD, "linalg"),
    "LU": (LU, "linalg"),
    "CREATION": (CREATION, None),
    "CREATION_ND": (CREATION_ND, "ndarray"),
    "IO": (IO, None),
    "IO_ND": (IO_ND, "ndarray"),
    "MATH": (MATH, None),
    "SEARCHING": (SEARCHING, None),
    "STATS": (STATS, None),
    "MISC": (MISC, None),
    "PACK": (PACK, None),
    "INDEX": (INDEX, None),
    "PAD": (PAD, None),
    "FUNCTIONAL": (FUNCTIONAL, None),
}


def check_function_exists(func_name: str, namespace_type: str | None) -> bool:
    """Check if a function exists in cuPyNumeric.

    Parameters
    ----------
    func_name : str
        Name of the function to check
    namespace_type : str | None
        Where to look: None (module-level), "ndarray", or "linalg"

    Returns
    -------
    bool
        True if function exists, False otherwise
    """
    if namespace_type == "ndarray":
        return hasattr(cn.ndarray, func_name)
    elif namespace_type == "linalg":
        return hasattr(cn.linalg, func_name)
    else:
        return hasattr(cn, func_name)


def get_all_functions_in_config():
    """Get all functions from Sphinx config with their namespace info.

    Returns
    -------
    dict
        Mapping of function name to namespace type
    """
    all_funcs = {}
    for category_name, (functions, namespace_type) in ALL_CATEGORIES.items():
        for func_name in functions:
            all_funcs[func_name] = namespace_type
    return all_funcs


# Fixtures
@pytest.fixture(scope="module")
def all_config_functions():
    """Get all functions from the Sphinx config."""
    return get_all_functions_in_config()


# Tests
@pytest.mark.parametrize("func_name", sorted(EXPECTED_UNIMPLEMENTED))
def test_expected_unimplemented_functions_are_missing(func_name):
    """Verify each function marked as NOT IMPLEMENTED is in config but missing from cuPyNumeric.

    This test ensures that:
    1. The function is listed in the Sphinx config (for documentation)
    2. The function does not exist in cuPyNumeric (will show "-" in docs)
    """
    all_funcs = get_all_functions_in_config()

    # Verify function IS in the config
    assert func_name in all_funcs, (
        f"Function '{func_name}' is in EXPECTED_UNIMPLEMENTED but missing from config! "
        f"Add it to the appropriate tuple in _comparison_config.py"
    )

    namespace_type = all_funcs[func_name]

    # Verify function does NOT exist in cuPyNumeric
    exists = check_function_exists(func_name, namespace_type)
    assert not exists, (
        f"Function '{func_name}' is marked as NOT IMPLEMENTED but exists in cuPyNumeric! "
        f"Either implement it properly or remove it from EXPECTED_UNIMPLEMENTED list."
    )


def test_all_remaining_config_functions_exist(all_config_functions):
    """Verify all functions still in Sphinx config are implemented."""
    missing = []

    for func_name, namespace_type in all_config_functions.items():
        # Skip functions we expect to be missing
        if func_name in EXPECTED_UNIMPLEMENTED:
            continue

        exists = check_function_exists(func_name, namespace_type)

        if not exists:
            missing.append(func_name)

    assert not missing, (
        f"Found {len(missing)} functions in Sphinx config that don't exist:\n"
        + "\n".join(f"  - {func}" for func in sorted(missing))
        + "\n\nThese should be added to EXPECTED_UNIMPLEMENTED and marked in config."
    )


def test_all_unimplemented_are_tracked(all_config_functions):
    """Verify all unimplemented functions are in EXPECTED_UNIMPLEMENTED.

    All functions that don't exist in cuPyNumeric should be tracked in
    the EXPECTED_UNIMPLEMENTED set.
    """
    # Find functions in config that are unimplemented but not tracked
    unexpected_missing = []

    for func_name, namespace_type in all_config_functions.items():
        exists = check_function_exists(func_name, namespace_type)

        if not exists and func_name not in EXPECTED_UNIMPLEMENTED:
            unexpected_missing.append(func_name)

    assert not unexpected_missing, (
        f"Found {len(unexpected_missing)} unimplemented functions not in EXPECTED_UNIMPLEMENTED:\n"
        + "\n".join(f"  - {func}" for func in sorted(unexpected_missing))
        + "\n\nAdd these to EXPECTED_UNIMPLEMENTED set in this test file."
    )


def test_no_false_positives_in_expected_list():
    """Verify EXPECTED_UNIMPLEMENTED doesn't contain implemented functions.

    This catches cases where a function was implemented but not removed
    from the expected unimplemented list.
    """
    all_funcs = get_all_functions_in_config()
    false_positives = []

    for func_name in EXPECTED_UNIMPLEMENTED:
        if func_name in all_funcs:  # Function is in config
            namespace_type = all_funcs[func_name]
            exists = check_function_exists(func_name, namespace_type)
            if exists:
                false_positives.append(func_name)

    assert not false_positives, (
        f"Found {len(false_positives)} functions in EXPECTED_UNIMPLEMENTED that actually exist:\n"
        + "\n".join(f"  - {func}" for func in sorted(false_positives))
        + "\n\nRemove these from EXPECTED_UNIMPLEMENTED list."
    )


def test_sphinx_config_consistency():
    """Comprehensive test ensuring Sphinx config is consistent with reality.

    This test verifies:
    1. Unimplemented functions are in the config (for docs) but not in cuPyNumeric
    2. Implemented functions are in both the config and cuPyNumeric
    3. The EXPECTED_UNIMPLEMENTED set matches actual unimplemented functions
    """
    all_funcs = get_all_functions_in_config()

    implemented = set()
    unimplemented = set()

    for func_name, namespace_type in all_funcs.items():
        exists = check_function_exists(func_name, namespace_type)
        if exists:
            implemented.add(func_name)
        else:
            unimplemented.add(func_name)

    # Verify unimplemented set matches expected
    assert unimplemented == EXPECTED_UNIMPLEMENTED, (
        f"Unimplemented function mismatch:\n"
        f"  Missing from EXPECTED_UNIMPLEMENTED: {sorted(unimplemented - EXPECTED_UNIMPLEMENTED)}\n"
        f"  Extra in EXPECTED_UNIMPLEMENTED: {sorted(EXPECTED_UNIMPLEMENTED - unimplemented)}"
    )

    # Verify no implemented functions are marked as unimplemented
    false_unimplemented = implemented & EXPECTED_UNIMPLEMENTED
    assert not false_unimplemented, (
        f"Found {len(false_unimplemented)} functions that exist but are in EXPECTED_UNIMPLEMENTED:\n"
        + "\n".join(f"  - {func}" for func in sorted(false_unimplemented))
        + "\n\nRemove these from EXPECTED_UNIMPLEMENTED."
    )


if __name__ == "__main__":
    # Allow running with: python test_unimplemented_functions.py
    import sys

    sys.exit(pytest.main(sys.argv))
