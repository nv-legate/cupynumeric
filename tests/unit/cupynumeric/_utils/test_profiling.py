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

import pytest

import cupynumeric as num
import cupynumeric._utils.profiling as profiling
from cupynumeric._utils.profiling import profiling_wrapper, _profiling_enabled
from cupynumeric._ufunc.ufunc import ufunc as cupynumeric_ufunc


class TestProfilingIntegration:
    """Test that profiling decorators are applied correctly"""

    def test_array_method_has_profiling(self) -> None:
        """Verify array methods have profiling wrapper"""
        arr = num.array([1, 2, 3])
        # Check that method has been wrapped (should have __wrapped__)
        assert hasattr(arr.sum, "__wrapped__")

    def test_module_function_has_profiling(self) -> None:
        """Verify module functions have profiling wrapper"""
        # iscomplex and isreal are module-level functions with profiling
        assert hasattr(num.iscomplex, "__wrapped__")
        assert hasattr(num.isreal, "__wrapped__")

    def test_ufunc_calls_have_profiling(self) -> None:
        """Verify direct ufunc calls have profiling wrappers"""
        profiled_ufuncs = {
            name
            for name, obj in vars(num).items()
            if isinstance(obj, cupynumeric_ufunc)
        }

        assert {"negative", "add", "frexp"} <= profiled_ufuncs
        for name in profiled_ufuncs:
            ufunc = getattr(num, name)
            assert hasattr(type(ufunc).__call__, "__wrapped__"), (
                f"{name} is missing a profiling wrapper"
            )


class TestProfilingWrapper:
    """Test profiling_wrapper function"""

    def test_profiling_wrapper_basic(self) -> None:
        """Test that profiling_wrapper wraps functions correctly"""

        def sample_func(x: int) -> int:
            return x * 2

        wrapped = profiling_wrapper(sample_func, "cupynumeric.sample_func")

        assert wrapped(5) == 10
        assert hasattr(wrapped, "__wrapped__")

    def test_profiling_wrapper_preserves_name(self) -> None:
        """Test that profiling_wrapper preserves function metadata"""

        def sample_func(x: int) -> int:
            """Sample docstring"""
            return x * 2

        wrapped = profiling_wrapper(sample_func, "cupynumeric.sample_func")

        assert wrapped.__doc__ == "Sample docstring"
        assert wrapped.__name__ == "sample_func"

    def test_profiling_wrapper_with_kwargs(self) -> None:
        """Test that profiling_wrapper handles kwargs"""

        def sample_func(x: int, y: int = 10) -> int:
            return x + y

        wrapped = profiling_wrapper(sample_func, "cupynumeric.sample_func")

        assert wrapped(5) == 15
        assert wrapped(5, y=20) == 25

    def test_profiling_enabled_checked_when_wrapping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that profiling mode is selected when the wrapper is built"""
        calls = 0

        def profiling_enabled() -> bool:
            nonlocal calls
            calls += 1
            return False

        def sample_func(x: int) -> int:
            return x * 2

        monkeypatch.setattr(profiling, "_profiling_enabled", profiling_enabled)

        wrapped = profiling_wrapper(sample_func, "cupynumeric.sample_func")

        assert calls == 1
        assert wrapped(5) == 10
        assert wrapped(6) == 12
        assert calls == 1
        assert hasattr(wrapped, "__wrapped__")

    def test_profiling_wrapper_with_dynamic_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that profiling_wrapper can resolve a name from call args"""
        names: list[str] = []

        class DummyProfileRange:
            def __init__(self, name: str) -> None:
                names.append(name)

            def __enter__(self) -> None:
                return None

            def __exit__(self, *exc_info: object) -> None:
                return None

        class Sample:
            profile_name = "sample"

        def sample_func(obj: Sample, value: int) -> int:
            return value

        monkeypatch.setattr(profiling, "_profiling_enabled", lambda: True)
        monkeypatch.setattr(profiling, "ProfileRange", DummyProfileRange)

        wrapped = profiling_wrapper(
            sample_func, lambda obj, _value: f"cupynumeric.{obj.profile_name}"
        )

        assert wrapped(Sample(), 5) == 5
        assert names == ["cupynumeric.sample"]


class TestProfilingEnabled:
    """Test profiling enabled check"""

    def test_profiling_enabled_cached(self) -> None:
        """Test that _profiling_enabled is cached"""
        # Call twice, should return same result (cached)
        result1 = _profiling_enabled()
        result2 = _profiling_enabled()

        assert result1 == result2
        assert isinstance(result1, bool)


class TestProfilingInfrastructure:
    """Test profiling infrastructure components"""

    def test_profiling_wrapper_preserves_exceptions(self) -> None:
        """Verify profiling_wrapper doesn't swallow exceptions"""

        def failing_func() -> None:
            raise ValueError("Test error")

        wrapped = profiling_wrapper(failing_func, "test.failing_func")

        with pytest.raises(ValueError, match="Test error"):
            wrapped()


class TestRandomProfiling:
    """Test that random Generator methods have profiling"""

    # List of ALL 36 methods that should have profiling
    PROFILED_METHODS = [
        "beta",
        "binomial",
        "bytes",
        "cauchy",
        "chisquare",
        "exponential",
        "f",
        "gamma",
        "geometric",
        "gumbel",
        "hypergeometric",
        "integers",
        "laplace",
        "logistic",
        "lognormal",
        "logseries",
        "negative_binomial",
        "noncentral_chisquare",
        "noncentral_f",
        "normal",
        "pareto",
        "poisson",
        "power",
        "random",
        "rayleigh",
        "standard_cauchy",
        "standard_exponential",
        "standard_gamma",
        "standard_t",
        "triangular",
        "uniform",
        "vonmises",
        "wald",
        "weibull",
        "zipf",
    ]

    @pytest.mark.parametrize("method_name", PROFILED_METHODS)
    def test_all_generator_methods_have_profiling(
        self, method_name: str
    ) -> None:
        """Verify all 36 Generator methods have profiling wrapper"""
        gen = num.random.default_rng(42)
        method = getattr(gen, method_name)
        assert hasattr(method, "__wrapped__"), (
            f"Method {method_name} is missing profiling wrapper"
        )


class TestUtilityFunctionProfiling:
    """Test that utility functions have profiling"""

    def test_iscomplex_has_profiling(self) -> None:
        """Verify iscomplex has profiling wrapper"""
        assert hasattr(num.iscomplex, "__wrapped__")

    def test_isreal_has_profiling(self) -> None:
        """Verify isreal has profiling wrapper"""
        assert hasattr(num.isreal, "__wrapped__")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
