#!/usr/bin/env bash
# Run the irm demo (build first: ./build.sh).
set -euo pipefail
cd "$(dirname "$0")"
java -cp out/main com.quant.irm.Demo
