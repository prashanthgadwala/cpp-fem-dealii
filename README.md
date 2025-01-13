#  Advanced Finite Element Analysis with deal.II

This project demonstrates the use of the deal.II library for finite element analysis. The deal.II library provides comprehensive tools for handling various aspects of finite element computations.

## Introduction

Using C++, the code is usually split into so-called header and source files, i.e., declaration and definition are separated. These files are then used to build a program via a compiler (translate code into computer language) and linker.

## Key Features

- **Triangulation**: Managing collections of cells and their geometric properties.
- **Finite Elements**: Describing finite element spaces and their properties.
- **Quadrature**: Defining quadrature points and weights on unit cells.
- **DoFHandler**: Managing degrees of freedom on triangulations.
- **Mapping**: Mapping points from unit cells to real cells.
- **FEValues**: Evaluating shape functions and their gradients at quadrature points.
- **Linear Systems**: Assembling and managing system matrices and vectors.
- **Linear Solvers**: Solving linear systems using iterative and direct solvers.
- **Output**: Generating output files for visualization.

## Documentation

For detailed documentation on the deal.II library, including tutorials and class references, visit the [deal.II documentation](https://dealii.org/).

## Usage

To use the deal.II library in your projects, ensure you have the appropriate version installed and refer to the official documentation for setup and usage instructions.

## Building the Program

The general workflow involves using CMake to generate Makefiles, which are then used by Make to build the program. The following steps outline this process:

1. **CMakeLists.txt**: This file contains the configuration for CMake.
    ```cmake
    # CMake script for the step-1 tutorial program:
    SET(TARGET "hello_world")
    SET(TARGET_SRC ${TARGET}.cc)
    SET(DEAL_II_DIR /home/dealiiuser/deal.II/)
    CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
    FIND_PACKAGE(deal.II 8.2 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../)
    IF(NOT ${deal.II_FOUND})
        MESSAGE(FATAL_ERROR "\n*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n")
    ENDIF()
    DEAL_II_INITIALIZE_CACHED_VARIABLES()
    PROJECT(${TARGET})
    DEAL_II_INVOKE_AUTOPILOT()
    ```

## Building and Running the Program

1. **Create a Build Directory**:
    - Open a terminal in the folder containing the CMakeLists.txt and source file.
    - Create a build directory and navigate into it:
        ```sh
        mkdir build
        cd build
        ```

2. **Load the deal.II Environment Variable**:
    - Load the deal.II environment variable:
        ```sh
        spack load dealii
        ```

3. **Generate Makefiles using CMake**:
    - Run CMake to generate the Makefiles:
        ```sh
        cmake .
        ```

4. **Compile the Program**:
    - Compile in debug mode:
        ```sh
        make debug
        ```

5. **Run the Program**:
    - Run the program:
        ```sh
        make run
        ```
