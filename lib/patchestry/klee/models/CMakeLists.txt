# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

# Compile C model files to LLVM bitcode using clang -emit-llvm.
# Bitcode files are linked into KLEE harnesses via --model-library, so the
# triple/datalayout must match the harness — KLEE itself runs in a Linux
# x86_64 docker image, and the verifier rejects model libraries whose
# triple+layout differ. Always target x86_64-unknown-linux-gnu regardless
# of the host build platform.

set(MODEL_SOURCES libc_models.c usb_hal_models.c)
set(MODEL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR})
set(MODEL_TARGET_TRIPLE x86_64-unknown-linux-gnu)

foreach(src ${MODEL_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    set(bc_file ${CMAKE_CURRENT_BINARY_DIR}/${name}.bc)
    add_custom_command(
        OUTPUT ${bc_file}
        COMMAND clang -emit-llvm -c -O0 -g -ffreestanding
            -target ${MODEL_TARGET_TRIPLE}
            -include ${MODEL_INCLUDE}/klee_stub.h
            -o ${bc_file}
            ${CMAKE_CURRENT_SOURCE_DIR}/${src}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${MODEL_INCLUDE}/klee_stub.h
        COMMENT "Compiling KLEE model ${src} -> ${name}.bc (${MODEL_TARGET_TRIPLE})"
    )
    list(APPEND MODEL_BC_FILES ${bc_file})
endforeach()

add_custom_target(klee-models ALL DEPENDS ${MODEL_BC_FILES})
