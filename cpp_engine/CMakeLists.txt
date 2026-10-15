cmake_minimum_required(VERSION 3.16)
project(TacticalCBMEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for AVX2 SIMD execution
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -mavx2 -Wall -Wextra)
endif()

include_directories(include)

# Target: Standalone Inference & Stage 5 Intervention CLI
add_executable(inference_cli src/inference_cli.cpp)

# Target: Preprocessor (for future native streaming)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/preprocessor.cpp")
    add_executable(preprocessor src/preprocessor.cpp)
endif()
