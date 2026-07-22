# Linux build glue for the bitnet_flutter_ai plugin.
#
# This is an FFI-only plugin: Flutter's plugin tool will compile the
# `bitnet_bridge` shared library produced by ../native/CMakeLists.txt
# and bundle it (plus libllama.so / libggml.so from the bitnet.cpp build)
# into the host application's lib/ directory automatically.
#
# Required: set the BITNET_LLAMA_PREFIX environment variable or pass
# -DBITNET_LLAMA_PREFIX=... on the cmake command line. It must point to a
# directory containing:
#   include/llama.h
#   lib/libllama.so
#   lib/libggml.so
#
# A staged prefix produced by symlinks works fine.

cmake_minimum_required(VERSION 3.13)

# The Flutter tool requires the project name to match the plugin name.
set(PROJECT_NAME "bitnet_flutter_ai")
project(${PROJECT_NAME} LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Locate the bitnet.cpp install prefix
# ---------------------------------------------------------------------------
if(NOT DEFINED BITNET_LLAMA_PREFIX)
    if(DEFINED ENV{BITNET_LLAMA_PREFIX})
        set(BITNET_LLAMA_PREFIX "$ENV{BITNET_LLAMA_PREFIX}")
    endif()
endif()

if(NOT BITNET_LLAMA_PREFIX OR NOT EXISTS "${BITNET_LLAMA_PREFIX}/include/llama.h")
    message(FATAL_ERROR
        "bitnet_flutter_ai: BITNET_LLAMA_PREFIX is not set or invalid.\n"
        "Set the BITNET_LLAMA_PREFIX environment variable to the directory "
        "containing include/llama.h and lib/libllama.so before running "
        "`flutter run -d linux` or `flutter build linux`.\n"
        "Got: '${BITNET_LLAMA_PREFIX}'")
endif()

message(STATUS "bitnet_flutter_ai: using BITNET_LLAMA_PREFIX=${BITNET_LLAMA_PREFIX}")

# Push the prefix into the variable that native/CMakeLists.txt reads BEFORE
# the subdirectory is added.
set(LLAMA_INSTALL_PREFIX "${BITNET_LLAMA_PREFIX}" CACHE PATH "" FORCE)

# ---------------------------------------------------------------------------
# Build the bridge from the native/ folder.
# ---------------------------------------------------------------------------
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../native"
                 "${CMAKE_CURRENT_BINARY_DIR}/native_build")

# ---------------------------------------------------------------------------
# Tell Flutter which shared libraries to copy into <bundle>/lib/.
# Resolve symlinks so the bundle contains real files, not dangling absolute
# links into the developer's bitnet.cpp checkout.
# ---------------------------------------------------------------------------
get_filename_component(LLAMA_BUNDLED_SO
    "${BITNET_LLAMA_PREFIX}/lib/libllama.so" REALPATH)
get_filename_component(GGML_BUNDLED_SO
    "${BITNET_LLAMA_PREFIX}/lib/libggml.so"  REALPATH)

if(NOT EXISTS "${LLAMA_BUNDLED_SO}")
    message(FATAL_ERROR "bitnet_flutter_ai: missing ${LLAMA_BUNDLED_SO}")
endif()
if(NOT EXISTS "${GGML_BUNDLED_SO}")
    message(FATAL_ERROR "bitnet_flutter_ai: missing ${GGML_BUNDLED_SO}")
endif()

# Bundle the bridge + its transitive llama/ggml libraries.
set(bitnet_flutter_ai_bundled_libraries
    "$<TARGET_FILE:bitnet_bridge>"
    "${LLAMA_BUNDLED_SO}"
    "${GGML_BUNDLED_SO}"
    PARENT_SCOPE
)
