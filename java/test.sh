#!/usr/bin/env bash
# Run the full JUnit 4 suite (build first: ./build.sh).
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

# Discover every *Test class from the compiled test tree.
CLASSES=$(cd out/test && find . -name '*Test.class' | sed 's|^\./||; s|\.class$||; s|/|.|g' | sort)

java -cp "out/main:out/test:${JUNIT}:${HAMCREST}" org.junit.runner.JUnitCore ${CLASSES}
