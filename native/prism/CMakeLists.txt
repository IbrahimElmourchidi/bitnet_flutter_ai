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

# Export only the bn_* C ABI. The exports list hides the fork's llama_/ggml_
# symbols and keeps bn_* exported at link time — so bn_* must NOT be compiled
# hidden (no -fvisibility=hidden), or they'd never reach the dynamic symbol
# table. Linux/Android use a GNU ld version script; Apple uses
# -exported_symbols_list; Windows a .def — added if that ships.
if(APPLE)
    # On iOS the two backends (bitnet static + prism) live in one app process
    # and both export bn_*. To avoid a collision, prism ships as its OWN dynamic
    # framework loaded by path via DynamicLibrary.open, keeping the fork's ggml
    # symbols private behind the exports list + two-level namespace. The bn_*
    # entry points ARE referenced by Dart FFI only (dlsym), never at link time,
    # so we must NOT dead-strip them: -exported_symbols_list keeps exactly the
    # seven bn_* and, being an export directive, marks them as used/retained.
    set_target_properties(prism_bridge PROPERTIES
        FRAMEWORK TRUE
        MACOSX_FRAMEWORK_IDENTIFIER "org.utanium.prismBridge"
        # Framework binary is named `prism_bridge` (no lib prefix).
        OUTPUT_NAME "prism_bridge"
        PREFIX ""
    )
    # -exported_symbols_list both restricts exports to the seven bn_* and marks
    # them as required roots, so they survive dead-stripping without extra flags.
    target_link_options(prism_bridge PRIVATE
        "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/prism_exports_apple.txt")
elseif(NOT WIN32)
    set_target_properties(prism_bridge PROPERTIES PREFIX "lib")
    target_link_options(prism_bridge PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/prism_exports.map")
else()
    set_target_properties(prism_bridge PROPERTIES PREFIX "lib")
endif()
