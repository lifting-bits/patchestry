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

// A custom `Varnode` used to represent the output of a `CALLOTHER` that
// we have invented.
public class DefinitionVarnode extends Varnode {
    private PcodeOp definitionPcodeOp;
    private HighVariable highVariable;

    public DefinitionVarnode(Address address, int size) {
        super(address, size);
    }

    public void setDef(HighVariable high, PcodeOp def) {
        this.definitionPcodeOp = def;
        this.highVariable = high;
    }

    @Override
    public PcodeOp getDef() {
        return this.definitionPcodeOp;
    }

    @Override
    public HighVariable getHigh() {
        return this.highVariable;
    }

    @Override
    public boolean isInput() {
        return false;
    }
};