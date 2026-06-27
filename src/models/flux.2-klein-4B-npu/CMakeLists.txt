cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++-13 CACHE STRING "C++ compiler")
project(host_flux_wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MODEL_NAME "flux.2-klein-4B" CACHE STRING "Executable model output folder name")
set(EXE_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../exe_models/${MODEL_NAME}"
    CACHE PATH "Directory for the generated model executable")
set(HOST_FLUX_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/libhost_flux.so"
    CACHE FILEPATH "Path to the prebuilt libhost_flux.so")

if(NOT EXISTS "${HOST_FLUX_LIB}")
    message(FATAL_ERROR "Missing ${HOST_FLUX_LIB}. Copy or build libhost_flux.so before configuring.")
endif()

add_library(host_flux_prebuilt SHARED IMPORTED GLOBAL)
set_target_properties(host_flux_prebuilt PROPERTIES
    IMPORTED_LOCATION "${HOST_FLUX_LIB}"
)

add_executable(host_flux_run host.cpp)
target_link_libraries(host_flux_run PRIVATE host_flux_prebuilt)

set_target_properties(host_flux_run PROPERTIES
    OUTPUT_NAME "run.exe"
    RUNTIME_OUTPUT_DIRECTORY "${EXE_MODEL_DIR}"
    BUILD_RPATH "$ORIGIN/../../lib"
    INSTALL_RPATH "$ORIGIN/../../lib"
)

add_custom_target(check
    COMMAND "${EXE_MODEL_DIR}/run.exe" --help
    DEPENDS host_flux_run
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
