# cuPyNumeric tests

This directory holds the cuPyNumeric test suites (`unit/`, `integration/`,
`array_api/`, …), run via [`test.py`](../test.py).

## Test marker policy: xfail / skip / skipif

Every `pytest` `xfail`, `skip`, and `skipif` marker under `tests/unit/` and
`tests/integration/` must carry a **classified reason**. This is enforced
automatically by the
`check-marker-reason` pre-commit hook
([scripts/hooks/check_marker_reason.py](../scripts/hooks/check_marker_reason.py))
so that known behavior differences are tracked, severity is explicit, and the
most severe gaps are linked to a tracking issue.

### Reason format

```
reason="[<severity>] <category>: <description> (<issue>)"
```

- `<severity>` (required): `severe` | `not-severe` | `triage`
- `<category>` (recommended): one of `wrong-result`, `hang`, `crash`,
  `unexpected-exception`, `not-implemented`, `exception-mismatch`,
  `dtype-divergence`, `gpu-only`, `numpy-version`, `other`
- `<issue>` (**required for `severe`**): a tracking issue, e.g.
  `cupynumeric.internal#601` or `#601`. Severe issues should be sub-issues of
  [#732](https://github.com/nv-legate/cupynumeric.internal/issues/732).

Examples:

```python
@pytest.mark.xfail(reason="[severe] wrong-result: amax over a tuple axis returns "
                          "wrong values (cupynumeric.internal#601)")

@pytest.mark.skipif(is_cpu_only, reason="[not-severe] gpu-only: requires cuSOLVER geev")

@pytest.mark.xfail(reason="[triage] needs manual classification")
```

Imperative `pytest.skip(...)` / `pytest.xfail(...)` calls only need a message
(no severity tag required) — they are typically dynamic environment guards.

### Severity

**`severe`** — cuPyNumeric can silently mislead or break the user. Requires a
linked issue.

- `wrong-result`: op completes but returns a wrong value / shape / dtype vs NumPy
- `hang`: deadlock / never returns
- `crash`: C++ abort (`LEGATE_ASSERT` / `LEGATE_CHECK` / `LEGATE_ABORT`) or segfault
- `unexpected-exception`: an exception that is *not* a deliberate "not supported"
  signal (e.g. an internal `AttributeError` / `struct.error`)

**`not-severe`** — known, benign divergence; the user is not misled.

- `not-implemented`: cuPyNumeric raises `NotImplementedError`
- `exception-mismatch`: both NumPy and cuPyNumeric fail, only the type/message differs
- `dtype-divergence`: different type-casting / promotion behavior
- `gpu-only` / `numpy-version`: legitimately skipped (cupy/GPU-only test on a
  CPU-only run; function missing in the current NumPy)

**`triage`** — not yet classified. Use sparingly and revisit; treat as a TODO.

### How enforcement works (the baseline ratchet)

The hook runs on every commit and in CI. The existing backlog of markers that
predate this policy is grandfathered in
[scripts/hooks/marker_reason_baseline.txt](../scripts/hooks/marker_reason_baseline.txt),
a **ratchet that may only shrink**. Enforcement is two layers:

- **pre-commit / normal mode** blocks a **new** non-compliant marker (or a new
  one of a *different kind* added to an already-listed function) that isn't in
  the working-tree baseline. Existing grandfathered markers are allowed until
  someone cleans them up.
- **CI `--against <merge-base>`** blocks growth of the baseline *file itself*.
  Without this, a PR could add a violation and pad the baseline in the same
  commit and sail through — the normal mode trusts the baseline it's handed.
  The CI step ([lint.yml](../.github/workflows/lint.yml)) compares the
  committed baseline against the merge-base with `main` and fails if any count
  grew or any entry appeared. Growth is **not possible through CI by design**;
  the rare intentional case (a key-format migration) must be landed with a
  deliberate, reviewed CI bypass.

### Updating the baseline

The baseline is **updated manually, by a human, and only ever shrinks.** CI and
pre-commit only *read* it — they never regenerate it (auto-regenerating would
silently bless new violations and defeat the whole point).

#### When do I need to update it?

| You did this | Update the baseline? |
|---|---|
| Removed an xfail (test now passes) | **Yes** — regenerate so the count drops |
| Gave an existing marker a compliant `[…]` reason | **Yes** — it's no longer a violation, regenerate |
| Added a **new** marker with a compliant reason | No — compliant markers are never in the baseline |
| Added a new marker with no/non-compliant reason | No — fix the reason instead; don't grandfather new debt |

**Forgetting to update is harmless** — it never errors or blocks a commit. The
baseline is an *upper bound* (allowed count), so removing a marker just leaves a
stale, unused entry (current count `0` ≤ allowed). The only downside: that stale
entry keeps a "slot" open, so a future non-compliant marker of the *same
violation kind* added to the *same function* would be allowed instead of
blocked. (The key includes the violation kind, so swapping in a *different*
kind of violation is still caught.) Regenerating keeps the ratchet tight, but
it's housekeeping, not a correctness requirement.

#### How to update it

After cleaning up one or more markers, regenerate and commit:

```bash
# regenerate (scans tests/unit and tests/integration; run from the repo root)
python scripts/hooks/check_marker_reason.py --update-baseline

# review the diff — numbers must only go DOWN, lines only disappear
git diff scripts/hooks/marker_reason_baseline.txt

git add scripts/hooks/marker_reason_baseline.txt
git commit -m "Shrink marker-reason baseline after cleaning <area>"
```

The single source of truth that you did it right: **the baseline diff only
shrinks** (counts decrease, entries are removed). If a count goes *up* or a new
entry appears, you introduced a new violation — fix it instead.

#### `--allow-grow` (rare)

`--update-baseline` **refuses to grow** the baseline by default. The only times
you should pass `--allow-grow`:

- **Initial creation / re-initialization** of the baseline.
- A deliberate, reviewed decision to defer a new marker (rare — prefer fixing
  the reason). Mention it in the PR so reviewers see the ratchet moved the wrong
  way on purpose.

```bash
python scripts/hooks/check_marker_reason.py --update-baseline --allow-grow
```

#### Baseline file format

`scripts/hooks/marker_reason_baseline.txt` is one entry per line:

```
<count>\t<relpath>::<qualname>::<marker-kind>::<violation-code>
```

Keyed by file + enclosing function (not line number), so it survives code
moving around. The trailing `<violation-code>` (e.g. `bare`, `missing-reason`,
`severe-no-issue`) distinguishes *kinds* of violation in the same scope, so
removing one and adding a structurally different one in the same function is
still caught. The end goal is to whittle it to **zero** and delete the file
plus this section.

### Roadmap

- A scheduled CI job to verify each `[severe]` issue is open and is a sub-issue
  of #732 (closed/completed → the xfail should be removed).
