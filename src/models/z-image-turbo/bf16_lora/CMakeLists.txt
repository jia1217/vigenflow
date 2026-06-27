cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++-13 CACHE STRING "C++ compiler")
project(host_zimage_bf16_lora_wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXE_MODEL_NAME "z-image-turbo/bf16" CACHE STRING "Executable model output folder name")
set(EXE_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../exe_models/${EXE_MODEL_NAME}"
    CACHE PATH "Directory for the generated model executable")
set(HOST_ZIMAGE_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/libhost_zimage_bf16_lora.so"
    CACHE FILEPATH "Path to the prebuilt libhost_zimage_bf16_lora.so")

if(NOT EXISTS "${HOST_ZIMAGE_LIB}")
    message(FATAL_ERROR "Missing ${HOST_ZIMAGE_LIB}. Copy or build libhost_zimage_bf16_lora.so before configuring.")
endif()

add_library(host_zimage_bf16_lora_prebuilt SHARED IMPORTED GLOBAL)
set_target_properties(host_zimage_bf16_lora_prebuilt PROPERTIES
    IMPORTED_LOCATION "${HOST_ZIMAGE_LIB}"
)

add_executable(host_zimage_bf16_lora_run host.cpp)
target_link_libraries(host_zimage_bf16_lora_run PRIVATE host_zimage_bf16_lora_prebuilt)

set_target_properties(host_zimage_bf16_lora_run PROPERTIES
    OUTPUT_NAME "run.exe"
    RUNTIME_OUTPUT_DIRECTORY "${EXE_MODEL_DIR}"
    BUILD_RPATH "$ORIGIN/../../../lib"
    INSTALL_RPATH "$ORIGIN/../../../lib"
)

add_custom_target(check
    COMMAND "${EXE_MODEL_DIR}/run.exe" --help
    DEPENDS host_zimage_bf16_lora_run
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
