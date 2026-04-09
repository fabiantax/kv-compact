cmake_minimum_required(VERSION 3.21)
project(kv-compact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KV_COMPACT_BUILD_TESTS  "Build unit tests" ON)
option(KV_COMPACT_BUILD_TOOL   "Build CLI tool (requires llama.cpp)" ON)
option(KV_COMPACT_BUILD_SHARED "Build shared library for Python bindings" ON)
option(KV_COMPACT_HIP          "Enable ROCm/HIP GPU acceleration (auto-detected if not set)" OFF)

# ============================================================================
# Platform auto-detection
# ============================================================================

# Auto-detect ROCm/HIP on Linux when not explicitly disabled
if(NOT KV_COMPACT_HIP AND NOT APPLE)
    find_program(_HIPCC hipcc PATHS /opt/rocm/bin ENV HIP_PATH)
    if(_HIPCC)
        set(KV_COMPACT_HIP ON)
        message(STATUS "Auto-detected ROCm/HIP (hipcc found at ${_HIPCC})")
    endif()
endif()

# Auto-detect GPU architecture via rocminfo
if(KV_COMPACT_HIP AND NOT DEFINED KV_COMPACT_HIP_ARCH)
    find_program(_ROCMINFO rocminfo PATHS /opt/rocm/bin)
    if(_ROCMINFO)
        execute_process(
            COMMAND ${_ROCMINFO}
            OUTPUT_VARIABLE _ROCMINFO_OUT
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        # Extract gfx architecture (e.g., gfx1151, gfx1100, gfx942)
        string(REGEX MATCH "gfx[0-9a-z]+" KV_COMPACT_HIP_ARCH "${_ROCMINFO_OUT}")
        if(KV_COMPACT_HIP_ARCH)
            message(STATUS "Auto-detected GPU architecture: ${KV_COMPACT_HIP_ARCH}")
        endif()
    endif()
    if(NOT KV_COMPACT_HIP_ARCH)
        set(KV_COMPACT_HIP_ARCH "gfx1151")
        message(STATUS "GPU architecture not detected, defaulting to ${KV_COMPACT_HIP_ARCH}")
    endif()
endif()

# ---- Header-only math library (no dependencies) ----
add_library(kv-compact-math INTERFACE)
target_include_directories(kv-compact-math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Optional ROCm/HIP acceleration ----
if(KV_COMPACT_HIP)
    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(rocblas REQUIRED)
    add_library(kv-compact-hip STATIC src/kv-compact-hip.hip)
    target_include_directories(kv-compact-hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(kv-compact-hip PUBLIC KV_COMPACT_HIP)
    target_link_libraries(kv-compact-hip PUBLIC roc::rocblas)
    set_target_properties(kv-compact-hip PROPERTIES HIP_ARCHITECTURES "${KV_COMPACT_HIP_ARCH}")
    message(STATUS "ROCm/HIP acceleration enabled (target: ${KV_COMPACT_HIP_ARCH})")
    message(STATUS "  rocBLAS: strided-batched GEMM for multi-head scoring + value refit")
endif()

# ---- C library API (standalone, no llama.cpp needed) ----
add_library(kv-compact-api STATIC src/kv-compact-api.cpp)
target_link_libraries(kv-compact-api PUBLIC kv-compact-math)
target_include_directories(kv-compact-api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# OpenMP — parallel chunk processing (optional, graceful fallback)
# ============================================================================
if(APPLE)
    set(OpenMP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../_deps/omp")  # unused but silences warnings
    # Homebrew libomp on Apple Silicon — find_package won't find it, set manually
    find_path(OMP_INCLUDE_DIR NAMES omp.h PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    find_library(OMP_LIBRARY NAMES omp libomp PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(OMP_INCLUDE_DIR AND OMP_LIBRARY)
        set(OpenMP_CXX_FOUND TRUE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${OMP_LIBRARY}")
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OMP_INCLUDE_DIR}"
            INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
            IMPORTED_LOCATION "${OMP_LIBRARY}"
        )
        message(STATUS "OpenMP found (Homebrew libomp) — parallel chunk processing enabled")
    else()
        set(OpenMP_CXX_FOUND FALSE)
        message(STATUS "OpenMP not found — install with: brew install libomp")
    endif()
else()
    find_package(OpenMP)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(kv-compact-api PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — parallel chunk processing enabled")
endif()

# ============================================================================
# BLAS/LAPACK — Cholesky least-squares solve
# ============================================================================
if(APPLE)
    # macOS: Apple Accelerate (always available)
    target_link_libraries(kv-compact-api PUBLIC "-framework Accelerate")
    target_compile_definitions(kv-compact-api PUBLIC
        KV_COMPACT_USE_BLAS=1
        ACCELERATE_NEW_LAPACK=1
        ACCELERATE_LAPACK_ILP64=1
    )
    message(STATUS "Apple Accelerate BLAS enabled — Cholesky LS solve (ILP64)")
else()
    # Linux: auto-detect OpenBLAS or system BLAS
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    if(BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(kv-compact-api PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        target_compile_definitions(kv-compact-api PUBLIC KV_COMPACT_USE_BLAS=1)
        message(STATUS "BLAS/LAPACK found — Cholesky LS solve enabled")
        message(STATUS "  BLAS: ${BLAS_LIBRARIES}")
    else()
        message(STATUS "BLAS/LAPACK not found — using fallback LS solver (install: apt install libopenblas-dev)")
    endif()
endif()

if(KV_COMPACT_HIP)
    target_link_libraries(kv-compact-api PUBLIC kv-compact-hip)
    target_compile_definitions(kv-compact-api PUBLIC KV_COMPACT_HIP)
endif()

# ============================================================================
# Shared library for Python/FFI bindings (enabled by default)
# ============================================================================
if(KV_COMPACT_BUILD_SHARED)
    add_library(kv-compact-shared SHARED src/kv-compact-api.cpp)
    target_link_libraries(kv-compact-shared PUBLIC kv-compact-math)
    target_include_directories(kv-compact-shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(kv-compact-shared PROPERTIES
        OUTPUT_NAME "kv-compact"
        POSITION_INDEPENDENT_CODE ON
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(kv-compact-shared PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(APPLE)
        target_link_libraries(kv-compact-shared PUBLIC "-framework Accelerate")
        target_compile_definitions(kv-compact-shared PUBLIC
            KV_COMPACT_USE_BLAS=1
            ACCELERATE_NEW_LAPACK=1
            ACCELERATE_LAPACK_ILP64=1
        )
    elseif(BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(kv-compact-shared PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        target_compile_definitions(kv-compact-shared PUBLIC KV_COMPACT_USE_BLAS=1)
    endif()
    if(KV_COMPACT_HIP)
        target_link_libraries(kv-compact-shared PUBLIC kv-compact-hip)
        target_compile_definitions(kv-compact-shared PUBLIC KV_COMPACT_HIP)
    endif()
    message(STATUS "Shared library enabled — python/kv_compact can use libkv-compact.so/.dylib")
endif()

# ---- Unit tests (standalone, no llama.cpp needed) ----
if(KV_COMPACT_BUILD_TESTS)
    add_executable(test-kv-compact-math tests/test-kv-compact-math.cpp)
    target_link_libraries(test-kv-compact-math PRIVATE kv-compact-math)

    add_executable(bench-kv-compact tests/bench-kv-compact.cpp)
    target_link_libraries(bench-kv-compact PRIVATE kv-compact-math)

    add_executable(test-kv-compact-api tests/test-kv-compact-api.cpp)
    target_link_libraries(test-kv-compact-api PRIVATE kv-compact-api)

    add_executable(bench-kv-compact-quality tests/bench-kv-compact-quality.cpp)
    target_link_libraries(bench-kv-compact-quality PRIVATE kv-compact-api)

    add_executable(bench-quant-throughput tests/bench-quant-throughput.cpp)
    target_link_libraries(bench-quant-throughput PRIVATE kv-compact-api)

    # Optimization validation benchmarks (paper-backed)
    add_executable(bench-opt-expected-attention tests/bench-opt-expected-attention.cpp)
    target_link_libraries(bench-opt-expected-attention PRIVATE kv-compact-api)

    add_executable(bench-opt-snapkv-window tests/bench-opt-snapkv-window.cpp)
    target_link_libraries(bench-opt-snapkv-window PRIVATE kv-compact-api)

    add_executable(bench-opt-value-norm-filter tests/bench-opt-value-norm-filter.cpp)
    target_link_libraries(bench-opt-value-norm-filter PRIVATE kv-compact-api)
endif()

# ---- CLI tool (needs llama.cpp) ----
if(KV_COMPACT_BUILD_TOOL)
    # Find llama.cpp - either via LLAMA_CPP_DIR or FetchContent
    if(LLAMA_CPP_DIR)
        # Use a local llama.cpp checkout
        message(STATUS "Using local llama.cpp: ${LLAMA_CPP_DIR}")
        set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
        add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
    else()
        # Fetch from GitHub
        include(FetchContent)
        set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            llama_cpp
            GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
            GIT_TAG        master
            GIT_SHALLOW    TRUE
        )
        message(STATUS "Fetching llama.cpp from GitHub (this may take a while)...")
        FetchContent_MakeAvailable(llama_cpp)
    endif()

    add_executable(llama-kv-compact src/kv-compact.cpp)
    target_link_libraries(llama-kv-compact PRIVATE
        kv-compact-math
        llama
        common
        ggml
    )

    # E2E test (also needs llama.cpp)
    add_executable(test-kv-compact-e2e tests/test-kv-compact-e2e.cpp)
    target_link_libraries(test-kv-compact-e2e PRIVATE
        kv-compact-math
        llama
        common
        ggml
    )

    # Real-model quality benchmark (thread-parallel per-layer compaction)
    find_package(Threads REQUIRED)
    add_executable(bench-kv-compact-model tests/bench-kv-compact-model.cpp)
    target_link_libraries(bench-kv-compact-model PRIVATE
        kv-compact-api
        llama
        common
        ggml
        Threads::Threads
    )

    # Code generation quality benchmark at scale
    add_executable(bench-kv-compact-codegen tests/bench-kv-compact-codegen.cpp)
    target_link_libraries(bench-kv-compact-codegen PRIVATE
        kv-compact-api
        llama
        common
        ggml
        Threads::Threads
    )

    # Real-model optimization validation gate
    add_executable(bench-opt-real-model tests/bench-opt-real-model.cpp)
    target_link_libraries(bench-opt-real-model PRIVATE
        kv-compact-api
        llama
        common
        ggml
        Threads::Threads
    )
endif()

# ============================================================================
# Configuration summary
# ============================================================================
message(STATUS "")
message(STATUS "=== kv-compact build configuration ===")
message(STATUS "  Platform:       ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_PROCESSOR})")
if(APPLE)
    message(STATUS "  Accelerator:    Apple Metal / Accelerate BLAS")
    message(STATUS "  Python target:  MLX (python/kv_compact/mlx.py)")
elseif(KV_COMPACT_HIP)
    message(STATUS "  Accelerator:    ROCm/HIP (${KV_COMPACT_HIP_ARCH})")
    message(STATUS "  Python target:  llama.cpp + HIP")
else()
    message(STATUS "  Accelerator:    CPU only")
endif()
message(STATUS "  OpenMP:         ${OpenMP_CXX_FOUND}")
if(BLAS_FOUND OR APPLE)
    message(STATUS "  BLAS:           YES")
else()
    message(STATUS "  BLAS:           NO (fallback solver)")
endif()
message(STATUS "  Shared library: ${KV_COMPACT_BUILD_SHARED}")
message(STATUS "  Tests:          ${KV_COMPACT_BUILD_TESTS}")
message(STATUS "  CLI tool:       ${KV_COMPACT_BUILD_TOOL}")
message(STATUS "")
if(APPLE)
    message(STATUS "  Quick start (macOS):")
    message(STATUS "    cd python && pip install -e '.[mlx]'")
    message(STATUS "    python examples/mlx_example.py -m mlx-community/Qwen3-8B-4bit")
elseif(KV_COMPACT_HIP)
    message(STATUS "  Quick start (ROCm):")
    message(STATUS "    cd python && pip install -e .")
    message(STATUS "    # Use libkv-compact.so via ctypes or llama-kv-compact CLI")
endif()
message(STATUS "")
