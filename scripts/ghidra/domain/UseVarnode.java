/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package domain;

import ghidra.program.model.pcode.Varnode;
import ghidra.program.model.address.Address;
import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.PcodeOp;

// A custome `Varnode` used to represent a rewritten input of a `PTRSUB`
// or other operation referencing a local variable that was not exactly
// correctly understood in the high p-code.
public class UseVarnode extends Varnode {
    private HighVariable highVariable;

    public UseVarnode(Address address, int size) {
        super(address, size);
    }

    public void setHighVariable(HighVariable high) {
        this.highVariable = high;
    }

    @Override
    public HighVariable getHigh() {
        return this.highVariable;
    }

    @Override
    public boolean isInput() {
        return true;
    }
};