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
# ruff: noqa: F403
from __future__ import annotations

from typing import Any
from warnings import warn

# The CLLR functions for the numpy module are broken up more or less according
# to the docs at https://numpy.org/doc/stable/reference/routines.html
#
# There are some discrepencies since some functions are repeated in multiple
# locations and since indexing routines are listed elsehwere for some reason.
# Sections/sub-modules that are currently missing are noted in comments.

# --- Array Creation Routines
# https://numpy.org/doc/stable/reference/routines.array-creation.html

from .creation_shape import *  # From shape or value
from .creation_data import *  # From existing data
from .creation_ranges import *  # Numerical Ranges
from .creation_matrices import *  # Building Matrices

# --- Array manipulation routines
# https://numpy.org/doc/stable/reference/routines.array-manipulation.html
#
# from .array_kind import *  # Changing kind of array
# from .array_add_remove import *  # Adding and removing elements

from .array_basic import *  # Basic operations
from .array_shape import *  # Changing array shape
from .array_transpose import *  # Transpose-like operations
from .array_dimension import *  # Changing number of dimensions
from .array_joining import *  # Joining arrays
from .array_splitting import *  # Splitting arrays
from .array_tiling import *  # Tiling arrays
from .array_rearrange import *  # Rearranging elements
from .array_delete import *  # Removing elements

# --- Binary operations
#  https://numpy.org/doc/stable/reference/routines.bitwise.html
#
# from .binary_elementwise_bit_ops import *  # Elementwise bit operations
# from .binary_output import *  # Output formatting

from .binary_bit_packing import *  # Bit packing

# --- Indexing routines
#
# These routines in the numpy module are a bit odd, they are documented under
# the array ref: https://numpy.org/doc/stable/reference/arrays.indexing.html

from .indexing import *

# --- Input and output
# https://numpy.org/doc/stable/reference/routines.io.html
#
# from .io_text import *  # Text files
# from .io_raw import *  # Raw binary files
# from .io_string import *  # String formatting
# from .io_memory import *  # Memory mapping files
# from .io_text import *  # Text formatting options
# from .io_base import *  # Base-n representations
# from .io_data import *  # Data sources
# from .io_binary import *  # Binary format description

from .io_numpy import *  # NumPy binary files (NPY, NPZ)

# --- Linear Algebra
# https://numpy.org/doc/stable/reference/routines.linalg.html

from .linalg_mvp import *  # Matrix and vector products

# --- Logic functions
# https://numpy.org/doc/stable/reference/routines.logic.html
#
# from .logic_ops import *  # Logical operations

from .logic_truth import *  # Truth value testing
from .logic_array_contents import *  # Array contents
from .logic_array_type import *  # Array type testing
from .logic_comparison import *  # Comparison

# --- Mathematical functions
# https://numpy.org/doc/stable/reference/routines.math.html
#
# from .math_trig import *  # Trigonometric functions
# from .math_hyp import *  # Hyperbolic functions
# from .math_exp_log import *  # Exponents and logarithms
# from .math_other import *  # Other special funtions
# from .math_floating import *  # Floating point routines
# from .math_arithmetic import *  # Arithmetic opertations

from .math_rounding import *  # Rounding
from .math_sum_prod_diff import *  # Sums, products, differences
from .math_complex import *  # Handling complex numbers
from .math_extrema import *  # Extrema finding
from .math_misc import *  # Miscellaneous

# --- Set routines
# https://numpy.org/doc/stable/reference/routines.set.html
#
# from .sets_boolean import *  # Boolean operations

from .sets_making import *  # Making proper sets

# --- Sorting, searching, and counting
# https://numpy.org/doc/stable/reference/routines.sort.html

from .ssc_sorting import *  # Sorting
from .ssc_searching import *  # Searching
from .ssc_counting import *  # Counting

# --- Statistics
# https://numpy.org/doc/stable/reference/routines.statistics.html
#

from .stats_order import *  # Order statistics
from .stats_avgs_vars import *  # Averages and variances
from .stats_correlating import *  # Correlating
from .stats_histograms import *  # Histograms

# --- Window functions
# https://numpy.org/doc/stable/reference/routines.window.html

from .window import *  # Various windows

# --- numpy.test (disabled)


def test(*args: Any, **kw: Any) -> None:
    warn(
        "cuPyNumeric cannot execute numpy.test() due to reliance "
        "on Numpy internals. For information about running the "
        "cuPyNumeric test suite, see: https://docs.nvidia.com/cupynumeric/latest/developer/index.html"
    )
