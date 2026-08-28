#!/bin/sh
# Build the irm C++ library, demo and tests.
#   tests: ctest --test-dir build --output-on-failure
set -e
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
