/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

 // Definition of ghidra pcode mnemonics

 #pragma once

 #define PCODE_MNEMONIC_LIST \
        X(COPY) \
        X(LOAD) \
        X(STORE) \
        X(BRANCH) \
        X(CBRANCH) \
        X(BRANCHIND) \
        X(CALL) \
        X(CALLIND) \
        X(USERDEFINED) \
        X(RETURN) \
        X(PIECE) \
        X(SUBPIECE) \
        X(INT_EQUAL) \
        X(INT_NOTEQUAL) \
        X(INT_LESS) \
        X(INT_SLESS) \
        X(INT_LESSEQUAL) \
        X(INT_SLESSEQUAL) \
        X(INT_ZEXT) \
        X(INT_SEXT) \
        X(INT_ADD) \
        X(INT_SUB) \
        X(INT_CARRY) \
        X(INT_SCARRY) \
        X(INT_SBORROW) \
        X(INT_2COMP) \
        X(INT_NEGATE) \
        X(INT_XOR) \
        X(INT_AND) \
        X(INT_OR) \
        X(INT_LEFT) \
        X(INT_RIGHT) \
        X(INT_SRIGHT) \
        X(INT_MULT) \
        X(INT_DIV) \
        X(INT_REM) \
        X(INT_SDIV) \
        X(INT_SREM) \
        X(BOOL_NEGATE) \
        X(BOOL_OR) \
        X(FLOAT_EQUAL) \
        X(FLOAT_NOTEQUAL) \
        X(FLOAT_LESS) \
        X(FLOAT_LESSEQUAL) \
        X(FLOAT_ADD) \
        X(FLOAT_SUB) \
        X(FLOAT_MULT) \
        X(FLOAT_DIV) \
        X(FLOAT_NEG) \
        X(FLOAT_ABS) \
        X(FLOAT_SQRT) \
        X(FLOAT_CEIL) \
        X(FLOAT_FLOOR) \
        X(FLOAT_ROUND) \
        X(FLOAT_NAN) \
        X(INT2FLOAT) \
        X(FLOAT2FLOAT) \
        X(TRUNC)

#define PCODE_VARNODE_TYPE \
        X(unique) \
        X(const) \
        X(register) \
        X(ram)