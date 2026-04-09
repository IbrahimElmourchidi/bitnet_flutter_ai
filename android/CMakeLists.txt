cmake_minimum_required(VERSION 3.22)
project(bitnet_bridge_android VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Paths — passed in by the Gradle externalNativeBuild block or manually:
#   cmake .. \
#     -DLLAMA_INCLUDE_DIR=/path/to/bitnet.cpp/include \
#     -DLLAMA_LIB=/path/to/bitnet.cpp/android/arm64-v8a/libllama.so
# ---------------------------------------------------------------------------
set(LLAMA_INCLUDE_DIR "" CACHE PATH "Directory containing llama.h")
set(LLAMA_LIB         "" CACHE FILEPATH "Path to libllama.so for the target ABI")

if(NOT LLAMA_INCLUDE_DIR OR NOT EXISTS "${LLAMA_INCLUDE_DIR}/llama.h")
    message(FATAL_ERROR "Set -DLLAMA_INCLUDE_DIR to a directory containing llama.h")
endif()
if(NOT LLAMA_LIB OR NOT EXISTS "${LLAMA_LIB}")
    message(FATAL_ERROR "Set -DLLAMA_LIB to the path of the prebuilt libllama.so")
endif()

# Prebuilt llama.so
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB}")

# bitnet_bridge.so — loaded at runtime by Dart DynamicLibrary.open()
add_library(bitnet_bridge SHARED
    ../native/bitnet_bridge.cc
)

target_include_directories(bitnet_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../native"   # bitnet_bridge.h
    "${LLAMA_INCLUDE_DIR}"                    # llama.h
)

target_link_libraries(bitnet_bridge PRIVATE
    llama
    android   # Android logging (optional but useful for debug builds)
    log
)

# Strip unexported symbols
target_link_options(bitnet_bridge PRIVATE
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../native/bitnet_bridge.map
)

set_target_properties(bitnet_bridge PROPERTIES
    OUTPUT_NAME "bitnet_bridge"
)
