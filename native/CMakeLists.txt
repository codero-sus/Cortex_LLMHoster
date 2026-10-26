cmake_minimum_required(VERSION 3.24)
project(CortexNativeGGUF VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The Cortex host links the inference core into its own process; it never
# launches llama-server. Release builds can opt into the separate upstream
# server executable so both runtime choices ship from the same pinned source.
option(CORTEX_BUILD_LLAMA_SERVER "Also build upstream llama-server for release bundles" OFF)
option(CORTEX_STATIC_MSVC_RUNTIME "Statically link the MSVC runtime for portable Windows bundles" OFF)
if(MSVC AND CORTEX_STATIC_MSVC_RUNTIME)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "Statically link the MSVC runtime" FORCE)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static native dependencies" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp's memory-fit and quantization support" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build upstream tests" FORCE)
set(LLAMA_BUILD_TOOLS ${CORTEX_BUILD_LLAMA_SERVER} CACHE BOOL "Build upstream tools only for release bundles" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build upstream examples" FORCE)
set(LLAMA_BUILD_SERVER ${CORTEX_BUILD_LLAMA_SERVER} CACHE BOOL "Build llama-server only for release bundles" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build the upstream unified app" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "Do not fetch or build a UI" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "Do not fetch a prebuilt UI" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "The private loopback servers do not need TLS" FORCE)
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "Optimize the CPU backend for the build machine")
endif()

# The native API is intentionally plain HTTP on loopback. Avoid pulling TLS or
# compression libraries into the private backend executable.
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "Disable HTTPS in the private backend" FORCE)
set(HTTPLIB_USE_WOLFSSL_IF_AVAILABLE OFF CACHE BOOL "Disable HTTPS in the private backend" FORCE)
set(HTTPLIB_USE_MBEDTLS_IF_AVAILABLE OFF CACHE BOOL "Disable HTTPS in the private backend" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "Disable compression in the private backend" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "Disable compression in the private backend" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "Disable compression in the private backend" FORCE)
set(HTTPLIB_INSTALL OFF CACHE BOOL "Do not install dependency targets" FORCE)
set(HTTPLIB_TEST OFF CACHE BOOL "Do not build dependency tests" FORCE)
set(JSON_BuildTests OFF CACHE BOOL "Do not build JSON dependency tests" FORCE)
set(JSON_Install OFF CACHE BOOL "Do not install JSON dependency targets" FORCE)

FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG 620d2225cc436ab0b001e3200e590e3b286ef9ab
  GIT_SHALLOW TRUE
)
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG 74ef3c42188acd6391f027c4337fb0e1a68c0024
  GIT_SHALLOW TRUE
)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama_cpp httplib nlohmann_json)

# Reuse the pinned quantizer without building the rest of llama.cpp's tools
# during ordinary local builds. Release mode enables LLAMA_BUILD_TOOLS above,
# which also creates llama-server and llama-quantize through upstream CMake.
if(NOT CORTEX_BUILD_LLAMA_SERVER)
  add_subdirectory(
    "${llama_cpp_SOURCE_DIR}/tools/quantize"
    "${llama_cpp_BINARY_DIR}/cortex-tools-quantize"
    EXCLUDE_FROM_ALL
  )
endif()

add_executable(cortex-gguf-host src/cortex_gguf_host.cpp)
target_include_directories(cortex-gguf-host PRIVATE "${llama_cpp_SOURCE_DIR}/common")
target_link_libraries(
  cortex-gguf-host
  PRIVATE llama llama-common httplib::httplib nlohmann_json::nlohmann_json Threads::Threads
)
set_target_properties(
  cortex-gguf-host
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_property(
    TARGET cortex-gguf-host
    PROPERTY RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/bin"
  )
endforeach()
set_target_properties(
  llama-quantize
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_property(
    TARGET llama-quantize
    PROPERTY RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/bin"
  )
endforeach()
if(TARGET llama-server)
  set_target_properties(
    llama-server
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_property(
      TARGET llama-server
      PROPERTY RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/bin"
    )
  endforeach()
endif()
if(CORTEX_BUILD_LLAMA_SERVER AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  foreach(runtime_target cortex-gguf-host llama-server llama-quantize)
    if(TARGET ${runtime_target})
      target_link_options(${runtime_target} PRIVATE -static-libgcc -static-libstdc++)
      set_target_properties(
        ${runtime_target}
        PROPERTIES
          BUILD_RPATH "$ORIGIN"
          BUILD_WITH_INSTALL_RPATH TRUE
          INSTALL_RPATH "$ORIGIN"
      )
    endif()
  endforeach()
endif()
target_compile_definitions(cortex-gguf-host PRIVATE NOMINMAX)
if(MSVC)
  target_compile_options(cortex-gguf-host PRIVATE /W4 /permissive-)
else()
  target_compile_options(cortex-gguf-host PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS cortex-gguf-host RUNTIME DESTINATION bin)
install(TARGETS llama-quantize RUNTIME DESTINATION bin OPTIONAL)
