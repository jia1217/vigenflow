cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++-13 CACHE STRING "C++ compiler")
project(host_flux2_klein_4B_edit_wrapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MODEL_NAME "flux2_klein_4B_edit" CACHE STRING "Executable model output folder name")
set(EXE_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../exe_models/${MODEL_NAME}"
    CACHE PATH "Directory for the generated model executable")
set(HOST_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/libhost_flux2_klein_4B_edit.so"
    CACHE FILEPATH "Path to the prebuilt libhost_flux2_klein_4B_edit.so")

if(NOT EXISTS "${HOST_LIB}")
    message(FATAL_ERROR "Missing ${HOST_LIB}. Copy or build libhost_flux2_klein_4B_edit.so before configuring.")
endif()

add_library(host_flux2_klein_4B_edit_prebuilt SHARED IMPORTED GLOBAL)
set_target_properties(host_flux2_klein_4B_edit_prebuilt PROPERTIES
    IMPORTED_LOCATION "${HOST_LIB}"
)

add_executable(host_flux2_klein_4B_edit_run host.cpp)
target_link_libraries(host_flux2_klein_4B_edit_run PRIVATE host_flux2_klein_4B_edit_prebuilt)

set_target_properties(host_flux2_klein_4B_edit_run PROPERTIES
    OUTPUT_NAME "run.exe"
    RUNTIME_OUTPUT_DIRECTORY "${EXE_MODEL_DIR}"
    BUILD_RPATH "$ORIGIN/../../lib"
    INSTALL_RPATH "$ORIGIN/../../lib"
)

add_custom_target(check
    COMMAND "${EXE_MODEL_DIR}/run.exe" --help
    DEPENDS host_flux2_klein_4B_edit_run
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
