# Builds libprism_bridge — prism_bridge.cc statically linked against PrismML's
# llama.cpp fork, as ONE self-contained shared library that exports only the
# bn_* C ABI. Hiding every other symbol (version script + hidden visibility) is
# essential: this library is loaded into the same process as the bitnet.cpp
# stack (libggml.so), and prism's modern ggml symbols must never interpose on
# it. Dart dlopen's each backend RTLD_LOCAL, so same-named symbols stay isolated.
cmake_minimum_required(VERSION 3.14)
project(prism_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED PRISM_LLAMA_CPP)
    message(FATAL_ERROR "Set -DPRISM_LLAMA_CPP=<path to PrismML/llama.cpp checkout>")
endif()

# Build the fork static, no extras, PIC (goes into a .so), no -march=native.
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON  OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL          OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE         OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${PRISM_LLAMA_CPP} prism_llama_build)

add_library(prism_bridge SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../prism_bridge.cc)

target_include_directories(prism_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PRISM_LLAMA_CPP}/include
    ${PRISM_LLAMA_CPP}/ggml/include
)

# `llama` transitively links ggml / ggml-base / ggml-cpu.
target_link_libraries(prism_bridge PRIVATE llama)

set_target_properties(prism_bridge PROPERTIES PREFIX "lib")

# Export only the bn_* C ABI (Linux/Android GNU ld). The version script both
# hides the fork's llama_/ggml_ symbols (local: *) and keeps bn_* exported
# (global:) at link time — so bn_* must NOT be compiled hidden (no
# -fvisibility=hidden), or they'd never reach the dynamic symbol table.
# macOS would use -exported_symbols_list; Windows a .def — added if those ship.
if(NOT APPLE AND NOT WIN32)
    target_link_options(prism_bridge PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/prism_exports.map")
endif()
