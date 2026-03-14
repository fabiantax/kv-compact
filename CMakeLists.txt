cmake_minimum_required(VERSION 3.21)
project(kv-compact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KV_COMPACT_BUILD_TESTS "Build unit tests" ON)
option(KV_COMPACT_BUILD_TOOL  "Build CLI tool (requires llama.cpp)" ON)
option(KV_COMPACT_ENABLE_PROFILING "Enable profiling instrumentation" OFF)

# ---- Header-only math library (no dependencies) ----
add_library(kv-compact-math INTERFACE)
target_include_directories(kv-compact-math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add profiling flags if enabled
if(KV_COMPACT_ENABLE_PROFILING)
    target_compile_definitions(kv-compact-math INTERFACE KV_COMPACT_PROFILING_ENABLED)
    message(STATUS "Profiling instrumentation enabled")
endif()

# ---- Unit tests (standalone, no llama.cpp needed) ----
if(KV_COMPACT_BUILD_TESTS)
    add_executable(test-kv-compact-math tests/test-kv-compact-math.cpp)
    target_link_libraries(test-kv-compact-math PRIVATE kv-compact-math)

    add_executable(bench-synthetic tests/bench-synthetic.cpp)
    target_link_libraries(bench-synthetic PRIVATE kv-compact-math)

    add_executable(test-kv-compact-adapter tests/test-kv-compact-adapter.cpp)
    target_link_libraries(test-kv-compact-adapter PRIVATE kv-compact-math)

    add_executable(bench-optimization tests/bench-optimization.cpp)
    target_link_libraries(bench-optimization PRIVATE kv-compact-math)
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
        FetchContent_Declare(
            llama_cpp
            GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
            GIT_TAG        6c770d16ca6870a9a419e37431f84f8b820df0c5
            PATCH_COMMAND  bash ${CMAKE_CURRENT_SOURCE_DIR}/patches/apply.sh <SOURCE_DIR>
        )
        message(STATUS "Fetching llama.cpp from GitHub (this may take a while)...")
        set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(llama_cpp)
    endif()

    add_executable(llama-kv-compact src/kv-compact.cpp)
    target_link_libraries(llama-kv-compact PRIVATE
        kv-compact-math
        llama
        common
        ggml
    )

    # Ensure console subsystem on Windows
    if(WIN32)
        set_target_properties(llama-kv-compact PROPERTIES
            LINK_FLAGS "/SUBSYSTEM:CONSOLE"
        )
    endif()

    # E2E test (also needs llama.cpp)
    add_executable(test-kv-compact-e2e tests/test-kv-compact-e2e.cpp)
    target_link_libraries(test-kv-compact-e2e PRIVATE
        kv-compact-math
        llama
        common
        ggml
    )

    # llama.cpp CLI tool (for speed testing)
    if(LLAMA_CPP_DIR)
        set(_llama_src_dir "${LLAMA_CPP_DIR}")
    elseif(llama_cpp_SOURCE_DIR)
        set(_llama_src_dir "${llama_cpp_SOURCE_DIR}")
    endif()
    if(_llama_src_dir AND EXISTS "${_llama_src_dir}/tools/cli/cli.cpp")
        add_executable(llama-cli ${_llama_src_dir}/tools/cli/cli.cpp)
        target_link_libraries(llama-cli PRIVATE llama common ggml)
        if(WIN32)
            set_target_properties(llama-cli PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
        endif()
    else()
        message(STATUS "Skipping llama-cli: cli.cpp not found at ${_llama_src_dir}/tools/cli/cli.cpp")
    endif()
endif()

# ---- Profiling target ----
if(KV_COMPACT_BUILD_TESTS AND KV_COMPACT_ENABLE_PROFILING)
    add_executable(profiled-kv-compact src/kv-compact.cpp)
    target_compile_definitions(profiled-kv-compact PRIVATE KV_COMPACT_PROFILING_ENABLED)
    target_link_libraries(profiled-kv-compact PRIVATE
        kv-compact-math
        llama
        common
        ggml
    )

    message(STATUS "Profiled executable: profiled-kv-compact")
endif()
