cmake_minimum_required(VERSION 3.22)
project(bitnet_bridge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# User-settable path to the bitnet.cpp / llama.cpp build output directory.
# Pass on the command line:
#   cmake .. -DLLAMA_INSTALL_PREFIX=/path/to/bitnet.cpp/build
# The directory must contain:
#   include/llama.h
#   lib/libllama.so  (Linux/Android)
#   lib/libllama.dylib (macOS)
#   lib/llama.lib + bin/llama.dll (Windows)
# ---------------------------------------------------------------------------
set(LLAMA_INSTALL_PREFIX "$ENV{BITNET_LLAMA_PREFIX}" CACHE PATH
    "Root of the bitnet.cpp build output (must contain include/ and lib/).")

if(NOT LLAMA_INSTALL_PREFIX AND NOT "$ENV{BITNET_LLAMA_PREFIX}" STREQUAL "")
    set(LLAMA_INSTALL_PREFIX "$ENV{BITNET_LLAMA_PREFIX}")
endif()
if(NOT LLAMA_INSTALL_PREFIX)
    message(FATAL_ERROR
        "LLAMA_INSTALL_PREFIX is not set.\n"
        "Run cmake with -DLLAMA_INSTALL_PREFIX=/path/to/bitnet.cpp/build")
endif()

# ---------------------------------------------------------------------------
# Find llama header and library
# ---------------------------------------------------------------------------
set(LLAMA_INCLUDE_DIR "${LLAMA_INSTALL_PREFIX}/include")
set(LLAMA_LIB_DIR     "${LLAMA_INSTALL_PREFIX}/lib")

if(NOT EXISTS "${LLAMA_INCLUDE_DIR}/llama.h")
    message(FATAL_ERROR "llama.h not found in ${LLAMA_INCLUDE_DIR}. "
        "Check LLAMA_INSTALL_PREFIX.")
endif()

# Resolve the correct library filename per platform
if(WIN32)
    set(LLAMA_LIB "${LLAMA_LIB_DIR}/llama.lib")
elseif(APPLE)
    set(LLAMA_LIB "${LLAMA_LIB_DIR}/libllama.dylib")
else()
    # Linux and Android
    set(LLAMA_LIB "${LLAMA_LIB_DIR}/libllama.so")
endif()

if(NOT EXISTS "${LLAMA_LIB}")
    message(FATAL_ERROR "llama library not found at ${LLAMA_LIB}. "
        "Check LLAMA_INSTALL_PREFIX.")
endif()

add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES
    IMPORTED_LOCATION "${LLAMA_LIB}"
)
if(WIN32)
    set_target_properties(llama PROPERTIES
        IMPORTED_IMPLIB "${LLAMA_LIB}"
    )
endif()

# ---------------------------------------------------------------------------
# bitnet_bridge shared library (loaded by Dart FFI at runtime)
# ---------------------------------------------------------------------------
add_library(bitnet_bridge SHARED
    bitnet_bridge.cc
)

target_include_directories(bitnet_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"   # bitnet_bridge.h
    "${LLAMA_INCLUDE_DIR}"          # llama.h
)

target_link_libraries(bitnet_bridge PRIVATE llama)

# Export only the bn_* symbols — keeps the ABI minimal
if(NOT WIN32)
    target_link_options(bitnet_bridge PRIVATE
        -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/bitnet_bridge.map
    )
endif()

# Platform-specific output naming.
# - On Linux/macOS the default "lib" prefix is correct (Dart's
#   DynamicLibrary.open('libbitnet_bridge.so') expects it).
# - On Windows the prefix must be empty so the DLL is named
#   `bitnet_bridge.dll` and not `libbitnet_bridge.dll`.
set_target_properties(bitnet_bridge PROPERTIES
    OUTPUT_NAME "bitnet_bridge"
)
if(WIN32)
    set_target_properties(bitnet_bridge PROPERTIES PREFIX "")
endif()

# When bundled into a Flutter desktop app, libbitnet_bridge.so sits in the
# same lib/ directory as libllama.so / libggml.so. Use $ORIGIN so the dynamic
# loader resolves the transitive dependencies regardless of the install path.
if(UNIX AND NOT APPLE)
    set_target_properties(bitnet_bridge PROPERTIES
        BUILD_RPATH   "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
elseif(APPLE)
    set_target_properties(bitnet_bridge PROPERTIES
        BUILD_RPATH   "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
endif()

# ---------------------------------------------------------------------------
# iOS: build as a static library instead of shared
# ---------------------------------------------------------------------------
if(IOS)
    add_library(bitnet_bridge_static STATIC
        bitnet_bridge.cc
    )
    target_include_directories(bitnet_bridge_static PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${LLAMA_INCLUDE_DIR}"
    )
    target_link_libraries(bitnet_bridge_static PRIVATE llama)
    set_target_properties(bitnet_bridge_static PROPERTIES
        OUTPUT_NAME "bitnet_bridge"
    )
endif()

# ---------------------------------------------------------------------------
# Install targets
# ---------------------------------------------------------------------------
install(TARGETS bitnet_bridge
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin   # Windows DLL
)
install(FILES bitnet_bridge.h DESTINATION include)
