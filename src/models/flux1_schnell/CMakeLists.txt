cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++-13 CACHE STRING "C++ compiler")
project(host_flux1_schnell_wrapper LANGUAGES CXX)

set(MODEL_NAME "flux1_schnell" CACHE STRING "Executable model output folder name")
include(../standalone_worker.cmake)
