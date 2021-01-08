cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(exemcl LANGUAGES CXX CUDA)
find_package(OpenMP)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

if (NOT OPENMP_FOUND)
    message(FATAL_ERROR "OpenMP not found. Aborting.")
endif ()

###################################################################
# OPTIONS
###################################################################
option(CREATE_TESTS "Should the test targets be created?" ON)
option(BUILD_NATIVE "Should native compilation be used?" ON)

###################################################################
# LIBRARIES
###################################################################

# Include Eigen
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen")

# Include pybind11
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11")

###################################################################
# TARGET CONFIGURATION
###################################################################

set(CMAKE_CUDA_ARCHITECTURES 53 60 61 62 70 72 75 80 86)  # all architectures to date with FP16 support.
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Check for native compilation.
if (BUILD_NATIVE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -march=native")
endif ()

# Allow calling host constexpr functions from device code (used to determine numeric_limits<T>::max).
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Add target.
pybind11_add_module(exemcl src/PythonModule.cu)
target_link_libraries(exemcl PRIVATE cublas OpenMP::OpenMP_CXX)
target_compile_options(exemcl PRIVATE -Xcompiler=-fopenmp)

# Create test targets, if requested.
if (CREATE_TESTS)
    set(GOOGLETEST_VERSION 1.10.0)
    set(gtest_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest/googletest")
    add_subdirectory(${gtest_src_dir})
    add_executable(exemcl-tests tests/UnitTests.cu)
    add_dependencies(exemcl-tests gtest gtest_main)
    target_include_directories(exemcl-tests PRIVATE ${gtest_src_dir}/include ${gtest_src_dir})
    target_compile_definitions(exemcl-tests PRIVATE EXEMCL_TESTFILES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/testfiles/")
    target_link_libraries(exemcl-tests gtest gtest_main cublas OpenMP::OpenMP_CXX)
    target_compile_options(exemcl-tests PRIVATE -forward-unknown-to-host-compiler -fopenmp)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        target_link_libraries(exemcl-tests gcov)
        target_compile_options(exemcl-tests PRIVATE -fprofile-arcs -ftest-coverage -fprofile-dir=${CMAKE_CURRENT_BINARY_DIR})
        target_link_options(exemcl-tests PRIVATE -lgcov --coverage -fprofile-arcs)
    endif ()
endif ()