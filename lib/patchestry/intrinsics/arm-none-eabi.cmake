include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Generic)

# Find the cross compiler.
find_program(CMAKE_C_COMPILER arm-none-eabi-gcc)
find_program(CMAKE_CXX_COMPILER arm-none-eabi-g++)
find_program(CMAKE_OBJCOPY arm-none-eabi-objcopy)

cmake_force_c_compiler(${CMAKE_C_COMPILER} GNU)
cmake_force_cxx_compiler(${CMAKE_CXX_COMPILER} GNU)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
