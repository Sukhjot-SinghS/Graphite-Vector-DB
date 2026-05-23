cmake_minimum_required(VERSION 3.15)
project(GraphiteEngine CXX)

# --- 1. Global Compiler Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Aggressive optimizations: O3, Native CPU Architecture (AVX2), and Position Independent Code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -fPIC")

# Tell CMake where our header files live
include_directories(${CMAKE_SOURCE_DIR}/core_engine)

# --- 2. TARGET 1: The Offline Builder ---
add_executable(builder core_engine/builder.cpp)

# --- 3. TARGET 2: The Pybind11 Live Engine ---
# Find Python first
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Bulletproof way to find Pybind11 even if installed via pip
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS "${pybind11_DIR}" NO_DEFAULT_PATH)

# Build the Python module (.so file)
pybind11_add_module(graphite_core core_engine/binding_linux_final.cpp)

# --- 4. The Auto-Mover ---
# Automatically move the compiled binaries to the root directory so Python can find them instantly
set_target_properties(graphite_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(builder PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})