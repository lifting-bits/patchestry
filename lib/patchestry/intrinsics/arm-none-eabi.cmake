# Toolchain file for arm-none-eabi

set(CMAKE_SYSTEM_NAME Generic)

# Find the cross compiler.
find_program(ARM_NONE_EABI_GCC arm-none-eabi-gcc)
find_program(ARM_NONE_EABI_GXX arm-none-eabi-g++)
find_program(CMAKE_OBJCOPY arm-none-eabi-objcopy)

set(CMAKE_C_COMPILER ${ARM_NONE_EABI_GCC})
set(CMAKE_CXX_COMPILER ${ARM_NONE_EABI_GXX})

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
