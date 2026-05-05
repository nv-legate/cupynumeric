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

from pathlib import Path

import pytest
from legate.util.fs import read_c_define
from legate.util.settings import EnvOnlySetting, PrioritizedSetting

import cupynumeric.settings as m

_expected_settings = (
    "disable_bounds_checking",
    "doctor_filename",
    "doctor_format",
    "doctor_traceback",
    "doctor",
    "fallback_stacktrace",
    "fast_math",
    "matmul_cache_size",
    "min_cpu_chunk",
    "min_gpu_chunk",
    "min_omp_chunk",
    "numpy_compat",
    "preload_cudalibs",
    "take_default",
    "test",
    "use_nccl_gather",
    "use_nccl_scatter",
    "warn",
)

_settings_with_test_defaults = (
    # skip fast math which uses getenv instead of extract_env
    # "fast_math",
    "min_gpu_chunk",
    "min_cpu_chunk",
    "min_omp_chunk",
    "matmul_cache_size",
)

ENV_HEADER = Path(__file__).parents[3] / "src" / "env_defaults.h"


class Test_convert_doctor_format:
    def test_plain(self) -> None:
        assert m.convert_doctor_format("plain") == "plain"
        assert m.convert_doctor_format("PLAIN") == "plain"

    def test_json(self) -> None:
        assert m.convert_doctor_format("json") == "json"
        assert m.convert_doctor_format("JSON") == "json"

    def test_csv(self) -> None:
        assert m.convert_doctor_format("csv") == "csv"
        assert m.convert_doctor_format("CSV") == "csv"

    def test_bad(self) -> None:
        with pytest.raises(
            ValueError, match="unknown cuPyNumeric Doctor format: junk"
        ):
            m.convert_doctor_format("junk")

    def test_type(self) -> None:
        assert (
            m.convert_doctor_format.type
            == 'DoctorFormat ("plain", "csv", or "json")'
        )


class Test_parse_bounds_checking:
    def test_all(self) -> None:
        assert m.parse_bounds_checking("all") == "all"
        assert m.parse_bounds_checking("ALL") == "all"

    def test_none(self) -> None:
        assert m.parse_bounds_checking("none") == "none"

    def test_selectors(self) -> None:
        assert (
            m.parse_bounds_checking("indexing,put,take") == "indexing,put,take"
        )

    def test_bad_selector(self) -> None:
        with pytest.raises(
            ValueError,
            match="unknown cuPyNumeric disabled bounds checking selector",
        ):
            m.parse_bounds_checking("junk")

    def test_bad_combination(self) -> None:
        with pytest.raises(
            ValueError,
            match='disabled bounds checking selectors "all" and "none" cannot be combined',
        ):
            m.parse_bounds_checking("all,indexing")

    def test_type(self) -> None:
        assert (
            m.parse_bounds_checking.type
            == 'DisableBoundsChecking ("none", "all", or comma-separated selectors: '
            "indexing, take, take_along_axis, put)"
        )


class TestSettings:
    def test_standard_settings(self) -> None:
        settings = [
            k
            for k, v in m.settings.__class__.__dict__.items()
            if isinstance(v, (PrioritizedSetting, EnvOnlySetting))
        ]
        assert set(settings) == set(_expected_settings)

    @pytest.mark.parametrize("name", _expected_settings)
    def test_prefix(self, name: str) -> None:
        ps = getattr(m.settings, name)
        assert (
            ps.env_var.startswith("CUPYNUMERIC_")
            or ps.env_var == "LEGATE_TEST"
        )

    def test_types(self) -> None:
        assert m.settings.doctor.convert_type == 'bool ("0" or "1")'
        assert (
            m.settings.disable_bounds_checking.convert_type
            == 'DisableBoundsChecking ("none", "all", or comma-separated selectors: '
            "indexing, take, take_along_axis, put)"
        )
        assert (
            m.settings.doctor_format.convert_type
            == 'DoctorFormat ("plain", "csv", or "json")'
        )
        assert m.settings.doctor_filename.convert_type == "str"
        assert m.settings.doctor_traceback.convert_type == 'bool ("0" or "1")'
        assert m.settings.preload_cudalibs.convert_type == 'bool ("0" or "1")'
        assert m.settings.use_nccl_gather.convert_type == 'bool ("0" or "1")'
        assert m.settings.use_nccl_scatter.convert_type == 'bool ("0" or "1")'
        assert m.settings.warn.convert_type == 'bool ("0" or "1")'
        assert (
            m.settings.fallback_stacktrace.convert_type == 'bool ("0" or "1")'
        )
        assert m.settings.numpy_compat.convert_type == 'bool ("0" or "1")'


class TestDefaults:
    def test_disable_bounds_checking(self) -> None:
        assert m.settings.disable_bounds_checking.default == "none"

    def test_doctor(self) -> None:
        assert m.settings.doctor.default is False

    def test_doctor_format(self) -> None:
        assert m.settings.doctor_format.default == "plain"

    def test_doctor_filename(self) -> None:
        assert m.settings.doctor_filename.default is None

    def test_doctor_traceback(self) -> None:
        assert m.settings.doctor_traceback.default is False

    def test_preload_cudalibs(self) -> None:
        assert m.settings.preload_cudalibs.default is False

    def test_use_nccl_gather(self) -> None:
        assert m.settings.use_nccl_gather.default is False

    def test_use_nccl_scatter(self) -> None:
        assert m.settings.use_nccl_scatter.default is False

    def test_warn(self) -> None:
        assert m.settings.warn.default is False

    def test_fallback_stacktrace(self) -> None:
        assert m.settings.fallback_stacktrace.default is False

    def test_numpy_compat(self) -> None:
        assert m.settings.numpy_compat.default is False

    @pytest.mark.skip(reason="Does not work in CI (path issue)")
    @pytest.mark.parametrize("name", _settings_with_test_defaults)
    def test_default(self, name: str) -> None:
        setting = getattr(m.settings, name)
        define = setting.env_var.removeprefix("CUPYNUMERIC_") + "_DEFAULT"
        expected = setting._convert(read_c_define(ENV_HEADER, define))
        assert setting.default == expected

    @pytest.mark.skip(reason="Does not work in CI (path issue)")
    @pytest.mark.parametrize("name", _settings_with_test_defaults)
    def test_test_default(self, name: str) -> None:
        setting = getattr(m.settings, name)
        define = setting.env_var.removeprefix("CUPYNUMERIC_") + "_TEST"
        expected = setting._convert(read_c_define(ENV_HEADER, define))
        assert setting.test_default == expected


class TestBoundsCheckEnabled:
    def teardown_method(self) -> None:
        m.settings.disable_bounds_checking.unset_value()

    def test_all(self) -> None:
        assert m.settings.bounds_check_enabled("indexing") is True
        assert m.settings.bounds_check_enabled("take") is True
        assert m.settings.bounds_check_enabled("put") is True

    def test_none(self) -> None:
        m.settings.disable_bounds_checking = "none"
        assert m.settings.bounds_check_enabled("indexing") is True
        assert m.settings.bounds_check_enabled("take") is True
        assert m.settings.bounds_check_enabled("put") is True

    def test_all_disabled(self) -> None:
        m.settings.disable_bounds_checking = "all"
        assert m.settings.bounds_check_enabled("indexing") is False
        assert m.settings.bounds_check_enabled("take") is False
        assert m.settings.bounds_check_enabled("put") is False

    def test_selective(self) -> None:
        m.settings.disable_bounds_checking = "take,put,indexing"
        assert m.settings.bounds_check_enabled("indexing") is False
        assert m.settings.bounds_check_enabled("take") is False
        assert m.settings.bounds_check_enabled("put") is False
        assert m.settings.bounds_check_enabled("take_along_axis") is True

    def test_bad_operation(self) -> None:
        with pytest.raises(
            ValueError, match="unknown bounds checking operation"
        ):
            m.settings.bounds_check_enabled("junk")  # type: ignore[arg-type]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
