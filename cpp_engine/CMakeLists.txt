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

# Option to link native ONNX Runtime C++ API for standalone GNN execution
option(ENABLE_ONNXRUNTIME "Link ONNX Runtime C++ API for full GNN inference" OFF)
set(ONNXRUNTIME_ROOTDIR "" CACHE PATH "Path to prebuilt ONNX Runtime package (e.g. onnxruntime-win-x64-1.20.1)")

# Target 1: Zero-Dependency Standalone Intervention CLI (Stage 5 Sub-Microsecond Engine)
add_executable(inference_cli src/inference_cli.cpp)

# Target 2: Native ONNX Runtime GNN Engine (Autonomous Full-Pipeline C++ Binary)
if(ENABLE_ONNXRUNTIME)
    if(NOT ONNXRUNTIME_ROOTDIR)
        message(FATAL_ERROR "ENABLE_ONNXRUNTIME is ON, but ONNXRUNTIME_ROOTDIR is not set. Pass -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime")
    endif()
    message(STATUS "Configuring Native ONNX Runtime Engine with root: ${ONNXRUNTIME_ROOTDIR}")
    add_executable(native_gnn_engine src/inference_cli.cpp)
    target_compile_definitions(native_gnn_engine PRIVATE ENABLE_ONNXRUNTIME=1)
    target_include_directories(native_gnn_engine PRIVATE ${ONNXRUNTIME_ROOTDIR}/include)
    target_link_directories(native_gnn_engine PRIVATE ${ONNXRUNTIME_ROOTDIR}/lib)
    if(WIN32)
        target_link_libraries(native_gnn_engine PRIVATE onnxruntime.lib)
        add_custom_command(TARGET native_gnn_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_newer
            ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:native_gnn_engine>
        )
    else()
        target_link_directories(native_gnn_engine PRIVATE ${ONNXRUNTIME_ROOTDIR}/lib)
        target_link_libraries(native_gnn_engine PRIVATE onnxruntime)
    endif()
endif()

# Target 3: Preprocessor (for future native streaming)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/preprocessor.cpp")
    add_executable(preprocessor src/preprocessor.cpp)
endif()
