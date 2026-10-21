cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++-13 CACHE STRING "C++ compiler")
project(host_flux2_klein_4B_wrapper LANGUAGES CXX)

set(MODEL_NAME "flux2_klein_4B" CACHE STRING "Executable model output folder name")
include(../standalone_worker.cmake)
