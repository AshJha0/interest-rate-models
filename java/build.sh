#!/usr/bin/env bash
# Build the irm Java library, demo and tests into out/.
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

rm -rf out
mkdir -p out/main out/test

# Library + demo (warnings are treated as errors to keep the build clean).
find src/main/java -name '*.java' | xargs javac -Xlint:all -Werror -d out/main

# Tests.
find src/test/java -name '*.java' | \
    xargs javac -Xlint:all -Werror -cp "out/main:${JUNIT}:${HAMCREST}" -d out/test

echo "build OK"
