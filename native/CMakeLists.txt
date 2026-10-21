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

# Compile the inference core into this process. No Python binding and no
# llama-server executable are used by the Cortex C++ runtime.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static native dependencies" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Do not build llama.cpp common/server layers" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build upstream tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build upstream tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build upstream examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama-server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build the upstream unified app" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "Do not fetch or build a UI" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "Do not fetch a prebuilt UI" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "The private loopback server does not need TLS" FORCE)
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

add_executable(cortex-gguf-host src/cortex_gguf_host.cpp)
target_link_libraries(
  cortex-gguf-host
  PRIVATE llama httplib::httplib nlohmann_json::nlohmann_json Threads::Threads
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
target_compile_definitions(cortex-gguf-host PRIVATE NOMINMAX)
if(MSVC)
  target_compile_options(cortex-gguf-host PRIVATE /W4 /permissive-)
else()
  target_compile_options(cortex-gguf-host PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS cortex-gguf-host RUNTIME DESTINATION bin)
