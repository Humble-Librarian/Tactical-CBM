@echo off
REM Compiles C++ geometry engine into WebAssembly via Emscripten (em++)
where em++ >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Emscripten compiler 'em++' not found in PATH.
    echo Please install or activate emsdk: https://emscripten.org/docs/getting_started/downloads.html
    exit /b 1
)

echo Compiling cpp_engine/src/wasm_bindings.cpp to dashboard/tactical_geometry.wasm ...
em++ -O3 -std=c++17 cpp_engine/src/wasm_bindings.cpp -Icpp_engine/include ^
    -s WASM=1 ^
    -s EXPORTED_FUNCTIONS="['_compute_tactical_concepts_wasm', '_malloc', '_free']" ^
    -s EXPORTED_RUNTIME_METHODS="['ccall', 'cwrap']" ^
    -o dashboard/tactical_geometry.js

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] WebAssembly module successfully compiled to dashboard/tactical_geometry.wasm
) else (
    echo [FAILED] Compilation failed.
    exit /b %ERRORLEVEL%
)
