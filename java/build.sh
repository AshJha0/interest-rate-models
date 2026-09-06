#!/usr/bin/env bash
# Build the irm Java library, demo and tests into out/.
# Override the JUnit 4 / Hamcrest jar locations with the JUNIT / HAMCREST
# environment variables if they are not at the Debian/Ubuntu default paths.
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=${JUNIT:-/usr/share/java/junit4.jar}
HAMCREST=${HAMCREST:-/usr/share/java/hamcrest.jar}

rm -rf out
mkdir -p out/main out/test

# Library + demo (warnings are treated as errors to keep the build clean).
find src/main/java -name '*.java' | xargs javac -Xlint:all -Werror -d out/main

# Tests.
find src/test/java -name '*.java' | \
    xargs javac -Xlint:all -Werror -cp "out/main:${JUNIT}:${HAMCREST}" -d out/test

echo "build OK"
