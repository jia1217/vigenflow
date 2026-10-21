cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++-13 CACHE STRING "C++ compiler")
project(host_qwen_image_2_1_turbo_edit_wrapper LANGUAGES CXX)

set(MODEL_NAME "qwen_image_2_1_turbo_edit" CACHE STRING "Executable model output folder name")
include(../standalone_worker.cmake)
