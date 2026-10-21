enable_language(ASM)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXE_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../exe_models/${MODEL_NAME}"
    CACHE PATH "Directory for the generated model executable")
set(HOST_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/libhost_${MODEL_NAME}.so"
    CACHE FILEPATH "Prebuilt model runtime embedded in the executable")
if(NOT EXISTS "${HOST_LIB}")
    message(FATAL_ERROR "Missing ${HOST_LIB}. Copy or build it before configuring.")
endif()
get_filename_component(HOST_LIB "${HOST_LIB}" ABSOLUTE)

set(EMBEDDED_HOST_ASM "${CMAKE_CURRENT_LIST_DIR}/embedded_host.S")
set_source_files_properties("${EMBEDDED_HOST_ASM}" PROPERTIES
    COMPILE_DEFINITIONS "VIGENFLOW_HOST_LIBRARY=\"${HOST_LIB}\""
    OBJECT_DEPENDS "${HOST_LIB}"
)
set(WORKER_TARGET "host_${MODEL_NAME}_run")
add_executable(${WORKER_TARGET} host.cpp
    "${CMAKE_CURRENT_LIST_DIR}/embedded_host.cpp" "${EMBEDDED_HOST_ASM}")
target_link_libraries(${WORKER_TARGET} PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(${WORKER_TARGET} PROPERTIES
    OUTPUT_NAME "run.exe"
    RUNTIME_OUTPUT_DIRECTORY "${EXE_MODEL_DIR}"
)
add_custom_target(check
    COMMAND "${EXE_MODEL_DIR}/run.exe" --help
    DEPENDS ${WORKER_TARGET}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
