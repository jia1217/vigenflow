cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++-13 CACHE STRING "C++ compiler")
project(host_krea2_turbo_wrapper LANGUAGES CXX)

set(MODEL_NAME "krea2_turbo" CACHE STRING "Executable model output folder name")
include(../standalone_worker.cmake)
