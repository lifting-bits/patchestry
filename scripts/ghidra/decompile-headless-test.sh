#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
SCRIPTS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
CMAKE_ARGS=
LLVM_DIR=$1

if [[ ! -d "${SCRIPTS_DIR}/test" ]]; then
    echo "No tests found!!!" && exit 1
fi

if [ -n "${LLVM_DIR}" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DLLVM_DIR=${LLVM_DIR}"
fi

cmake ${CMAKE_ARGS} -B ${SCRIPTS_DIR}/build -S ${SCRIPTS_DIR}/test
cmake --build ${SCRIPTS_DIR}/build
ctest  --output-on-failure  --test-dir ${SCRIPTS_DIR}/build
