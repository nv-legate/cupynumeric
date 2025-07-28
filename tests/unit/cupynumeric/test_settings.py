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
    "doctor_filename",
    "doctor_format",
    "doctor_traceback",
    "doctor",
    "fallback_stacktrace",
    "fast_math",
    "force_thunk",
    "matmul_cache_size",
    "max_eager_volume",
    "min_cpu_chunk",
    "min_gpu_chunk",
    "min_omp_chunk",
    "numpy_compat",
    "preload_cudalibs",
    "report_coverage",
    "report_dump_callstack",
    "report_dump_csv",
    "test",
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
            m.settings.doctor_format.convert_type
            == 'DoctorFormat ("plain", "csv", or "json")'
        )
        assert m.settings.doctor_filename.convert_type == "str"
        assert m.settings.doctor_traceback.convert_type == 'bool ("0" or "1")'
        assert m.settings.preload_cudalibs.convert_type == 'bool ("0" or "1")'
        assert m.settings.warn.convert_type == 'bool ("0" or "1")'
        assert m.settings.report_coverage.convert_type == 'bool ("0" or "1")'
        assert (
            m.settings.report_dump_callstack.convert_type
            == 'bool ("0" or "1")'
        )
        assert m.settings.report_dump_csv.convert_type == "str"
        assert (
            m.settings.fallback_stacktrace.convert_type == 'bool ("0" or "1")'
        )
        assert m.settings.numpy_compat.convert_type == 'bool ("0" or "1")'


class TestDefaults:
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

    def test_warn(self) -> None:
        assert m.settings.warn.default is False

    def test_report_coverage(self) -> None:
        assert m.settings.report_coverage.default is False

    def test_report_dump_callstack(self) -> None:
        assert m.settings.report_dump_callstack.default is False

    def test_report_dump_csv(self) -> None:
        assert m.settings.report_dump_csv.default is None

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
