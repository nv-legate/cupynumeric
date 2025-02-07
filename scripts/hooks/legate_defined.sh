#!/usr/bin/env bash
output=$(
  grep -E \
       -n \
       -H \
       -C 1 \
       --color=always \
       -e '#\s*if[n]?def\s+LEGATE_\w+' \
       -e '#(\s*if\s+)?[!]?defined\s*\(\s*LEGATE_\w+' \
       -e '#.*defined\s*\(\s*LEGATE_\w+' \
       -e '#\s*elif\s+LEGATE_\w+' \
       -- \
       "$@"
      )
rc=$?
if [[ ${rc} -eq 1 ]]; then
  # no matches found, that's a good thing
  exit 0
elif [[ ${rc} -eq 0 ]]; then
  echo "x ===------------------------------------------------------------------=== x"
  echo "${output}"
  echo ""
  echo "Instances of preprocessor ifdef/ifndef/if defined found, use"
  echo "LEGATE_DEFINED() instead:"
  echo ""
  echo "- #ifdef LEGATE_USE_FOO"
  echo "- #include \"foo.h\""
  echo "- #endif"
  echo "+ #if LEGATE_DEFINED(LEGATE_USE_FOO)"
  echo "+ #include \"foo.h\""
  echo "+ #endif"
  echo ""
  echo "- #ifdef LEGATE_USE_FOO"
  echo "- x = 2;"
  echo "- #endif"
  echo "+ if (LEGATE_DEFINED(LEGATE_USE_FOO)) {"
  echo "+   x = 2;"
  echo "+ }"
  echo "x ===------------------------------------------------------------------=== x"
  exit 1
else
  exit ${rc}
fi
