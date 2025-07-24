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

from dataclasses import dataclass
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    MethodType,
)

import numpy

__all__ = ("GROUPED_CONFIGS", "MISSING_NP_REFS", "NUMPY_CONFIGS", "SKIP")

SKIP = {
    "abs",
    "add_docstring",
    "add_newdoc_ufunc",
    "add_newdoc",
    "alen",
    "alltrue",
    "bitwise_not",
    "compare_chararrays",
    "cumproduct",
    "fastCopyAndTranspose",
    "get_array_wrap",
    "iterable",
    "max",
    "min",
    "product",
    "recfromcsv",
    "recfromtxt",
    "round",
    "safe_eval",
    "set_numeric_ops",
    "size",
    "sometrue",
    "test",
    "Tester",
}

# these do not have valid intersphinx references
MISSING_NP_REFS = {
    "numpy.loads",
    "numpy.mafromtxt",
    "numpy.ndarray.flat",
    "numpy.ndarray.shape",
    "numpy.ndarray.T",
    "numpy.ndfromtxt",
}


@dataclass(frozen=True)
class SectionConfig:
    title: str
    attr: str | None
    types: tuple[type, ...] | None = None
    names: tuple[str, ...] | None = None


# numpy 1.25 introduced private _ArrayFunctionDispatcher, handle gently
FUNCTIONS = (FunctionType, BuiltinFunctionType, type(numpy.broadcast))
METHODS = (MethodType, MethodDescriptorType)
UFUNCS = (numpy.ufunc,)

NUMPY_CONFIGS = [
    SectionConfig("Module-Level", None),
    SectionConfig("Multi-Dimensional Array", "ndarray"),
    SectionConfig("Linear Algebra", "linalg"),
    SectionConfig("Discrete Fourier Transform", "fft"),
    SectionConfig("Random Sampling", "random"),
]

CONVOLVE = ("convolve", "correlate")

LOGICAL = (
    "all",
    "allclose",
    "any",
    "array_equal",
    "array_equiv",
    "isclose",
    "iscomplex",
    "iscomplexobj",
    "isfortran",
    "isneginf",
    "isposinf",
    "isreal",
    "isrealobj",
    "isscalar",
)

EINSUM = ("dot", "einsum", "inner", "outer", "tensordot", "trace")

SET = (
    "in1d",
    "intersect1d",
    "isin",
    "setdiff1d",
    "setxor1d",
    "union1d",
    "unique",
)

MANIP = (
    "append",
    "array_split",
    "asanyarray",
    "asarray_chkfinite",
    "asarray",
    "ascontiguousarray",
    "asfortranarray",
    "asmatrix",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block",
    "broadcast_arrays",
    "broadcast_to",
    "column_stack",
    "concatenate",
    "copyto",
    "delete",
    "dsplit",
    "dstack",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "insert",
    "moveaxis",
    "ravel",
    "repeat",
    "require",
    "reshape",
    "resize",
    "roll",
    "rollaxis",
    "rot90",
    "row_stack",
    "shape",
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "tile",
    "transpose",
    "trim_zeros",
    "vsplit",
    "vstack",
)


MANIP_ND = ("flatten",)

FACTOR = ("cholesky", "qr")

SVD = ("lstsq", "matrix_rank", "pinv", "svd")

EIGEN = ("eig", "eigh", "eigvals", "eigvalsh")

LU = ("det", "inv", "slogdet", "solve", "tensorinv", "tensorsolve")

CREATION = (
    "arange",
    "array",
    "bmat",
    "copy",
    "diag",
    "diagflat",
    "empty_like",
    "empty",
    "eye",
    "frombuffer",
    "fromfile",
    "fromfunction",
    "fromiter",
    "full_like",
    "full",
    "geomspace",
    "identity",
    "linspace",
    "logspace",
    "meshgrid",
    "ones_like",
    "ones",
    "tri",
    "tril",
    "triu",
    "vander",
    "zeros_like",
    "zeros",
)

CREATION_ND = ("copy",)

IO = (
    "array_repr",
    "array_str",
    "array2string",
    "base_repr",
    "binary_repr",
    "format_float_positional",
    "format_float_scientific",
    "fromregex",
    "fromstring",
    "genfromtxt",
    "get_printoptions",
    "load",
    "loadtxt",
    "printoptions",
    "save",
    "savetxt",
    "savez_compressed",
    "savez",
    "set_printoptions",
)

IO_ND = ("tofile", "tolist")

MATH = (
    "amax",
    "amin",
    "angle",
    "around",
    "clip",
    "cross",
    "cumprod",
    "cumsum",
    "diff",
    "ediff1d",
    "fix",
    "gradient",
    "i0",
    "imag",
    "interp",
    "nan_to_num",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmin",
    "nanprod",
    "nansum",
    "prod",
    "real_if_close",
    "real",
    "sinc",
    "sum",
    "trapz",
    "unwrap",
)

SEARCHING = (
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "argwhere",
    "count_nonzero",
    "extract",
    "flatnonzero",
    "lexsort",
    "nanargmax",
    "nanargmin",
    "nonzero",
    "partition",
    "searchsorted",
    "sort_complex",
    "sort",
    "where",
)

STATS = (
    "average",
    "bincount",
    "corrcoef",
    "cov",
    "digitize",
    "histogram_bin_edges",
    "histogram",
    "histogram2d",
    "histogramdd",
    "mean",
    "median",
    "nanmean",
    "nanmedian",
    "nanpercentile",
    "nanquantile",
    "nanstd",
    "nanvar",
    "percentile",
    "ptp",
    "quantile",
    "std",
    "var",
)

MISC = ("kron", "ndim")

PACK = ("packbits", "unpackbits")

INDEX = (
    "choose",
    "compress",
    "diag_indices_from",
    "diag_indices",
    "diagonal",
    "fill_diagonal",
    "indices",
    "ix_",
    "mask_indices",
    "nested_iters",
    "place",
    "put_along_axis",
    "put",
    "putmask",
    "ravel_multi_index",
    "select",
    "take_along_axis",
    "take",
    "tril_indices_from",
    "tril_indices",
    "triu_indices_from",
    "triu_indices",
    "unravel_index",
)

PAD = ("pad",)

FUNCTIONAL = ("apply_along_axis", "apply_over_axes", "frompyfunc", "piecewise")

GROUPED_CONFIGS = [
    SectionConfig("Convolve and Correlate", None, names=CONVOLVE),
    SectionConfig("Ufuncs", None, UFUNCS),
    SectionConfig("Logical operations", None, names=LOGICAL),
    SectionConfig("Einsum and related", None, names=EINSUM),
    SectionConfig("Discrete Fourier transform", "fft", types=FUNCTIONS),
    SectionConfig("Set operations", None, names=SET),
    SectionConfig("Array manipulation", None, names=MANIP),
    SectionConfig("Array manipulation (ndarray)", "ndarray", names=MANIP_ND),
    SectionConfig("Factorizations", "linalg", names=FACTOR),
    SectionConfig("SVD and related", "linalg", names=SVD),
    SectionConfig("Eigenvalues", "linalg", names=EIGEN),
    SectionConfig("LU factorization and related", "linalg", names=LU),
    SectionConfig("Input and output", None, names=IO),
    SectionConfig("Input and output (ndarray)", "ndarray", names=IO_ND),
    SectionConfig("Array creation", None, names=CREATION),
    SectionConfig("Array creation (ndarray)", "ndarray", names=CREATION_ND),
    SectionConfig("Mathematical functions", None, names=MATH),
    SectionConfig("Searching, sorting, and counting", None, names=SEARCHING),
    SectionConfig("Advanced statistics", None, names=STATS),
    SectionConfig("Miscellaneous matrix routines", None, names=MISC),
    SectionConfig("Packing and unpacking bits", None, names=PACK),
    SectionConfig("Indexing", None, names=INDEX),
    SectionConfig("Padding arrays", None, names=PAD),
    SectionConfig("Random sampling", "random", types=FUNCTIONS),
    SectionConfig("Functional programming", None, names=FUNCTIONAL),
]
