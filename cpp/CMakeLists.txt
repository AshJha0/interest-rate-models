cmake_minimum_required(VERSION 3.16)
project(irm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

# Absolute path to the shared data directory (../data relative to cpp/):
# quote CSVs and golden values.
set(IRM_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")

# ---- library ---------------------------------------------------------------
add_library(irm
    src/mathutils.cpp
    src/rootfind.cpp
    src/curve.cpp
    src/bootstrap.cpp
    src/quotes.cpp
    src/optimize.cpp
    src/vasicek.cpp
    src/hullwhite.cpp
)
target_include_directories(irm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- demo ------------------------------------------------------------------
add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE irm)
target_compile_definitions(demo PRIVATE IRM_DATA_DIR="${IRM_DATA_DIR}")

# ---- tests -----------------------------------------------------------------
enable_testing()
find_package(GTest REQUIRED)

add_executable(irm_tests
    tests/test_rootfind.cpp
    tests/test_optimize.cpp
    tests/test_curve.cpp
    tests/test_bootstrap.cpp
    tests/test_vasicek.cpp
    tests/test_hullwhite.cpp
    tests/test_golden.cpp
)
target_link_libraries(irm_tests PRIVATE irm GTest::gtest GTest::gtest_main)
target_compile_definitions(irm_tests PRIVATE IRM_DATA_DIR="${IRM_DATA_DIR}")

include(GoogleTest)
gtest_discover_tests(irm_tests)
