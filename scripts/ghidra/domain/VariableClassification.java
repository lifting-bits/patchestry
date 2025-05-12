/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package domain;

public enum VariableClassification {
    UNKNOWN,
    PARAMETER,
    LOCAL,
    NAMED_TEMPORARY,
    TEMPORARY,
    GLOBAL,
    FUNCTION,
    CONSTANT
};