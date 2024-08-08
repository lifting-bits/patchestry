#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
set -e
SCRIPTS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

INPUT_FILE=$1
FUNCTION=$2
PATTERN_FILE=$3
tmpdir=/tmp

# It expects decompile-headless.sh will be in its parent directory.
${SCRIPTS_DIR}/../decompile-headless.sh ${INPUT_FILE} ${FUNCTION} ${tmpdir}
FileCheck ${PATTERN_FILE} --input-file ${tmpdir}/patchestry.out.json || exit 1
