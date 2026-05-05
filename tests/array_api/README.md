# Array API Test Baseline

This directory records the local, opt-in baseline for running the upstream
`data-apis/array-api-tests` suite against cuPyNumeric. It is not part of the
regular cuPyNumeric CI suite, and it is not a conformance claim.

The current baseline is intentionally a scaffold for follow-up work. Expect a
skip-heavy summary because broad expected-failure groups are applied as skips
by default, and optional extensions are excluded from the default run.

## Setup

Use a cuPyNumeric development environment that can import the current checkout.
A disposable virtual environment is recommended because the local constraints
pin pytest, Hypothesis, and the upstream suite's other test dependencies. The
constraints file is named for the Array API standard version under test, while
the upstream test-suite checkout is pinned separately:

```sh
python scripts/run_array_api_tests.py --setup-only
python -m pip install \
  -r .tmp/array-api-tests/2026.02.26/requirements.txt \
  -c tests/array_api/constraints-2025.12.txt
```

The runner creates `.tmp/array-api-tests/2026.02.26` if it does not exist, pins
it to upstream tag `2026.02.26`, verifies commit
`41379d15d26d67a1e66c840e775d41a8a7fb1516`, and initializes the upstream
`array-api` submodule.
If checkout setup is interrupted, remove that `.tmp/array-api-tests/2026.02.26`
directory and rerun the setup command.

The upstream `array-api-tests` checkout and the Array API standard version are
separate pins. This baseline runs the `data-apis/array-api-tests` suite from
tag `2026.02.26`, but configures that suite to test the Array API `2025.12`
spec by setting `ARRAY_API_TESTS_VERSION=2025.12`.

`constraints-2025.12.txt` pins only the upstream test runner dependencies, such
as pytest and Hypothesis. It does not pin cuPyNumeric or Legate; run the
baseline from the cuPyNumeric development environment you want to test. The
runner checks those test dependency versions before invoking pytest.

When using a CUDA-enabled Legate build on a host without GPUs, run with an
explicit CPU configuration, for example:

```sh
LEGATE_AUTO_CONFIG=0 LEGATE_CONFIG="--cpus 4 --gpus 0 --sysmem 4000" \
  python scripts/run_array_api_tests.py --max-examples 5
```

## Running

```sh
python scripts/run_array_api_tests.py --max-examples 5
```

By default the runner tests only the core Array API surface. It passes
`--disable-extension linalg fft` to upstream pytest and applies
`skips-2025.12-core.txt` for optional-extension name checks that the pinned
upstream suite does not mark as extension tests.

Use `--include-extensions` to remove those default extension skips. The normal
baseline xfail file still applies in that mode, including the broad
`array_api_tests/test_linalg.py` entry described below. Use
`--run-expected-failures` or a narrower `--xfails-file` when actively auditing
tests currently covered by expected-failure groups.

Entries in `xfails-2025.12.txt` are applied as skips by default via
`ARRAY_API_TESTS_XFAIL_MARK=skip`. Some entries guard tests that can abort
before pytest reports an xfail in the current baseline. Use
`--run-expected-failures` for local audits; the runner sets
`ARRAY_API_TESTS_XFAIL_MARK=xfail` in that mode even if the caller's
environment requests skips, and enables strict XPASS handling.

Extra pytest options and filters can be passed after `--`, for example:

```sh
python scripts/run_array_api_tests.py --max-examples 5 -- -k test_has_names
python scripts/run_array_api_tests.py --max-examples 5 -- --disable-deadline
```

Use `--pytest-target` for positional pytest selectors:

```sh
python scripts/run_array_api_tests.py \
  --pytest-target array_api_tests/test_has_names.py \
  -- -k __array_namespace_info__
```

## Failure Taxonomy

The expected-failure file is grouped by follow-up issue. Keep new entries as
narrow upstream node-id substrings where practical, and avoid broad module
xfails unless the category is intentionally deferred here.

### Standard Wrappers And Docs: #1800

The upstream suite uses Array API spelling and signatures. This group tracks
missing or incompatible standard namespace wrappers, keyword-only behavior,
and documented standard exports that are not yet provided under the Array API
names.
The primary Hypothesis function modules are currently grouped at module
granularity because upstream value generation and comparison paths exercise
scalar conversion and truthiness hooks before individual wrapper gaps can be
reported reliably.

`array_api_tests/test_linalg.py` is a broad entry in this group because the
pinned suite mixes required top-level linear algebra tests with optional
extension tests. With extension tests disabled, the current local audit still
aborts in `test_matmul` while Hypothesis materializes array inputs.

### Array Object Hooks: #1801

Array object tests cover standard properties, methods, and operators beyond the
current namespace-dispatch smoke tests. Direct name and signature checks for
`device`, `to_device`, and `mT` are expected to pass. The broad object-suite
entry guards native aborts observed while auditing object-level tests such as
`test_getitem`.

### Remaining Unsupported Behavior

Any xfail that does not fit the named follow-up issues belongs here with
a comment describing the current unsupported behavior. This is intentionally a
short-term holding area; prefer filing a narrower follow-up issue when a stable
root cause emerges.
