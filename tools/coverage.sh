#!/bin/bash
mkdir report
gcov ../cmake-build-debug/*.gcda
gcovr -g -r ../ -e ../lib -e ../src/CudaHelpers.cu --html-details report/report.html
