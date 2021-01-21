# Exemplar-based clustering for GPUs [![Read Manual](https://img.shields.io/badge/read-manual-informational)](https://philippjh.github.io/exemcl/)

This repository provides an CUDA implementation of the *Exemplar-based clustering* submodular function. This GPU algorithm has originally been discussed in:

> TBA.

The algorithm can both be used from C++ and Python. More information on this package, including a quick start guide and how to use this within C++, is
given [here](https://philippjh.github.io/exemcl/).

## Install from source

### Requirements:

- A CUDA compliant GPU with compute capability 5.3 or newer (NVIDIA Pascal architecture or newer).
- NVCC >= 11.2.67 (older versions may work, however, at least CUDA 11 is required since C++17 compiler support is **mandatory**.)
- CMake >= 3.13
- OpenMP

### Building the Python library

- Recursively clone this repository by issuing `git clone --recurse-submodules git@github.com:philippjh/exemcl.git`
- Change your working directory to the root of the repository. Run `pip install .`
- The Python package manager will now build and install the package.

## Running the test suite

This package provides two test suites: The first test suite confirms correct operation of the library within Python and writes a set of test files to disk. The second test suite
confirms correct operation of the library within C++ and is intended to provide extensive information on code coverage. It is **required** to run the Python test suite prior to the
C++ test suite since latter relies on those test files, which were written to disk by the Python test suite. Hence, to run both test suites follow these steps:

- Recursively clone this repository by issuing `git clone --recurse-submodules git@github.com:philippjh/exemcl.git`
- Change your working directory to the root of the repository. Run `python3 tests/UnitTests.py`. Verify, that a `testfiles` directory has been written to the `tests` folder.
- *(Optional, if you want to run the C++ test suite)*: Create a build directory, e.g. by issuing `mkdir build` from the repository root directory and `cd` into that folder.
- Create the required Makefiles by running `cmake -DCMAKE_BUILD_TYPE=Release ..`. Replace `Release` with `Debug` if you are interested in coverage data.
- Run `make`. Once building has finished, you can perform the C++ tests by running the `exemcl-tests` executable.

## Acknowledgments

Part of the work on this paper has been supported by Deutsche Forschungsgemeinschaft (DFG) within the Collaborative Research Center SFB 876 "Providing Information by
Resource-Constrained Analysis", project A1, http://sfb876.tu-dortmund.de and by the German Competence Center for Machine Learning Rhine Ruhr
(ML2R, https://www.ml2r.de, 01IS18038A), funded by the German Federal Ministry for Education and Research.
