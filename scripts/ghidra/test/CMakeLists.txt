cmake_minimum_required(VERSION 3.25)

include(CTest)

SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "-g")

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.c")

# Iterate over each source file and compile them
foreach(SOURCE_FILE ${SOURCES})
  get_filename_component(BINARY_NAME ${SOURCE_FILE} NAME_WE)
  add_executable(${BINARY_NAME} ${SOURCE_FILE})
  file(READ "${CMAKE_SOURCE_DIR}/src/${BINARY_NAME}.function" FUNCTION)
  add_test(NAME ${BINARY_NAME}Test
    COMMAND ${CMAKE_SOURCE_DIR}/../decompile-headless.sh ${CMAKE_CURRENT_BINARY_DIR}/${BINARY_NAME} ${FUNCTION} /tmp &&
            [ -s /tmp/patchestry.out.json ] || exit 1
  )
endforeach()
