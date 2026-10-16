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
    message(STATUS "Configuring Native ONNX Runtime GNN Engine...")
    
    if(NOT ONNXRUNTIME_ROOTDIR)
        set(ORT_VERSION "1.20.1")
        include(FetchContent)
        
        if(WIN32)
            set(ORT_ARCH_NAME "onnxruntime-win-x64-${ORT_VERSION}")
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCH_NAME}.zip")
        elseif(UNIX AND NOT APPLE)
            set(ORT_ARCH_NAME "onnxruntime-linux-x64-${ORT_VERSION}")
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCH_NAME}.tgz")
        else()
            message(FATAL_ERROR "Automated FetchContent for ONNX Runtime is supported on Windows x64 and Linux x64 only.")
        endif()
        
        message(STATUS "Fetching prebuilt ONNX Runtime ${ORT_VERSION} from ${ORT_URL}")
        FetchContent_Declare(
            onnxruntime_prebuilt
            URL ${ORT_URL}
        )
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        set(ORT_INCLUDE_DIR "${onnxruntime_prebuilt_SOURCE_DIR}/include")
        set(ORT_LIB_DIR "${onnxruntime_prebuilt_SOURCE_DIR}/lib")
    else()
        message(STATUS "Using user-specified ONNX Runtime root: ${ONNXRUNTIME_ROOTDIR}")
        set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include")
        set(ORT_LIB_DIR "${ONNXRUNTIME_ROOTDIR}/lib")
    endif()

    add_executable(native_gnn_engine src/inference_cli.cpp)
    target_compile_definitions(native_gnn_engine PRIVATE ENABLE_ONNXRUNTIME=1)
    target_include_directories(native_gnn_engine PRIVATE ${ORT_INCLUDE_DIR})
    target_link_directories(native_gnn_engine PRIVATE ${ORT_LIB_DIR})

    if(WIN32)
        target_link_libraries(native_gnn_engine PRIVATE onnxruntime)
        add_custom_command(TARGET native_gnn_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_LIB_DIR}/onnxruntime.dll"
            $<TARGET_FILE_DIR:native_gnn_engine>
        )
    else()
        target_link_libraries(native_gnn_engine PRIVATE onnxruntime pthread dl)
        set_target_properties(native_gnn_engine PROPERTIES
            BUILD_RPATH "${ORT_LIB_DIR}"
            INSTALL_RPATH "${ORT_LIB_DIR}"
        )
    endif()
endif()

# Target 3: Preprocessor (for future native streaming)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/preprocessor.cpp")
    add_executable(preprocessor src/preprocessor.cpp)
endif()
