/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package domain;

import ghidra.program.model.address.Address;
import ghidra.program.model.data.DataType;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighOther;
import ghidra.program.model.pcode.Varnode;

// A manually-created temporary variable with a single use.
public class HighTemporary extends HighOther {
    public HighTemporary(DataType dataType, Varnode varnode, Varnode[] inst, Address programCounter, HighFunction highFunction) {
        super(dataType, varnode, inst, programCounter, highFunction);
    }
}