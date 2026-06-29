# Copyright 2026 NVIDIA Corporation
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

from typing import Any, FrozenSet, Sequence

from legate.core.utils import OrderedSet

from ..runtime import runtime

# cuFFT uses optimized mixed-radix kernels only for transform lengths whose
# prime factors are all <= 131. A larger prime factor forces the Bluestein
# algorithm, which is much slower and can require several times more GPU
# scratch memory (a large per-call work area).
CUFFT_MAX_EFFICIENT_PRIME = 131
_CUFFT_SMALL_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
)


def has_large_prime_factor(n: int) -> bool:
    """Whether ``n`` has a prime factor > ``CUFFT_MAX_EFFICIENT_PRIME``.

    Only the primes <= 131 are trial-divided out, so the cost is bounded by a
    fixed, small number of divisions regardless of the magnitude of ``n`` (we
    never need the exact largest factor, only whether one exceeds 131).
    """
    n = int(n)
    if n < 2:
        return False
    for p in _CUFFT_SMALL_PRIMES:
        while n % p == 0:
            n //= p
        if n == 1:
            # Fully factored by primes <= 131: no large prime factor.
            return False
    # A factor > 131 survived after dividing out every prime <= 131.
    return True


def warn_if_bluestein_fft(
    axes: Sequence[int], in_store: Any, out_store: Any
) -> FrozenSet[int]:
    """Detect (and warn about) transform lengths that hit the Bluestein path.

    A transformed-axis length with a prime factor > ``CUFFT_MAX_EFFICIENT_PRIME``
    makes cuFFT fall back to the Bluestein algorithm, which is both much slower
    and far more memory-hungry than a mixed-radix transform.
    """
    in_shape = tuple(in_store.shape)
    out_shape = tuple(out_store.shape)
    slow = []
    for ax in OrderedSet(axes):
        # cuFFT plans for the larger of the in/out extents (R2C/C2R differ).
        length = int(max(in_shape[ax], out_shape[ax]))
        if has_large_prime_factor(length):
            slow.append((ax, length))
    if not slow:
        return frozenset()
    details = ", ".join(f"axis {ax} (length {length})" for ax, length in slow)
    runtime.warn(
        f"cuPyNumeric is computing an FFT over {details} whose length has a "
        f"prime factor > {CUFFT_MAX_EFFICIENT_PRIME}, so cuFFT falls back to "
        "the Bluestein algorithm. You may notice significantly decreased "
        "performance and much higher GPU memory usage for this function call. "
        "Zero-padding the transformed axis to a length whose prime factors "
        f"are all <= {CUFFT_MAX_EFFICIENT_PRIME} (e.g. the next power of two) "
        "avoids this.",
        category=RuntimeWarning,
    )
    return frozenset(ax for ax, _ in slow)
