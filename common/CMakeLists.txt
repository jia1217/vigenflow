cmake_minimum_required(VERSION 3.28)






find_package(Boost REQUIRED)
find_package(XRT REQUIRED)
# print boost and xrt dir
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "XRT_INCLUDE_DIRS: ${XRT_INCLUDE_DIRS}")

set(NPU_UTILS_SRCS
    ${HOME_DIR}/common/npu_utils.cpp
    ${HOME_DIR}/common/npu_instr_utils.cpp
)

set(NPU_UTILS_HEADERS
    ${HOME_DIR}/common/npu_utils.hpp
    ${HOME_DIR}/common/vector_view.hpp
    ${HOME_DIR}/common/debug_utils.hpp
    ${HOME_DIR}/common/npu_instr_utils.hpp
)


file(GLOB INSTR_UTILS_HEADERS "${HOME_DIR}/common/instr_utils/*.hpp")
list(APPEND NPU_UTILS_HEADERS ${INSTR_UTILS_HEADERS})



# === Step 1: Define library ===
add_library(npu_common STATIC ${NPU_UTILS_SRCS})

target_include_directories(npu_common PUBLIC
    ${HOME_DIR}/common
    ${XRT_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# target_link_libraries(npu_common PUBLIC xrt)
# target_link_libraries(npu_common PUBLIC boost_program_options)