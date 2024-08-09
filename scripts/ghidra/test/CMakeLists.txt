cmake_minimum_required(VERSION 3.25)

include(CTest)

SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "-g")

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.c")

function(get_function_name filename function)
    file(READ "${filename}" content)
    string(FIND "${file_content}" "\n" newline)

    # Extract the first line
    if(newline GREATER -1)
        string(SUBSTRING "${content}" 0 "${newline}" first_line)
    else()
        set(first_line "${content}")
    endif()
    set(${function} "${first_line}" PARENT_SCOPE)
endfunction()

# Iterate over each source file and compile them
foreach(SOURCE_FILE ${SOURCES})
  get_filename_component(BINARY_NAME ${SOURCE_FILE} NAME_WE)
  add_executable(${BINARY_NAME} ${SOURCE_FILE})

  get_function_name("${CMAKE_SOURCE_DIR}/src/${BINARY_NAME}.function" FUNCTION)
  add_test(NAME ${BINARY_NAME}Test
    COMMAND ${CMAKE_SOURCE_DIR}/run-test.sh ${CMAKE_CURRENT_BINARY_DIR}/${BINARY_NAME} ${FUNCTION} ${SOURCE_FILE}
  )
endforeach()
