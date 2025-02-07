#!/bin/bash

if [ $# -eq 0 ]
  then
    REALM_BACKTRACE=1 LEGATE_TEST=1 python run.py
elif [ $# -eq 1 ]  && [ "$1" = "ctest" ]
  then
    echo "Using ctest"
    cd build
    REALM_BACKTRACE=1 LEGATE_TEST=1 LEGION_DEFAULT_ARGS="-ll:cpu 4" ctest --output-on-failure "$@"
else
    echo "Invalid arguments"
fi
