cmake_minimum_required(VERSION 3.21)
project(kv-compact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KV_COMPACT_BUILD_TESTS "Build unit tests" ON)
option(KV_COMPACT_BUILD_TOOL  "Build CLI tool (requires llama.cpp)" ON)
option(KV_COMPACT_HIP         "Enable ROCm/HIP GPU acceleration" OFF)

# ---- Header-only math library (no dependencies) ----
add_library(kv-compact-math INTERFACE)
target_include_directories(kv-compact-math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Optional ROCm/HIP acceleration ----
if(KV_COMPACT_HIP)
    enable_language(HIP)
    find_package(hip REQUIRED)
    add_library(kv-compact-hip STATIC src/kv-compact-hip.hip)
    target_include_directories(kv-compact-hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(kv-compact-hip PUBLIC KV_COMPACT_HIP)
    set_target_properties(kv-compact-hip PROPERTIES HIP_ARCHITECTURES "gfx1151")
    message(STATUS "ROCm/HIP acceleration enabled (target: gfx1151 / RDNA 3.5)")
endif()

# ---- C library API (standalone, no llama.cpp needed) ----
add_library(kv-compact-api STATIC src/kv-compact-api.cpp)
target_link_libraries(kv-compact-api PUBLIC kv-compact-math)
target_include_directories(kv-compact-api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(KV_COMPACT_HIP)
    target_link_libraries(kv-compact-api PUBLIC kv-compact-hip)
    target_compile_definitions(kv-compact-api PUBLIC KV_COMPACT_HIP)
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

    # Real-model quality benchmark
    add_executable(bench-kv-compact-model tests/bench-kv-compact-model.cpp)
    target_link_libraries(bench-kv-compact-model PRIVATE
        kv-compact-api
        llama
        common
        ggml
    )
endif()
