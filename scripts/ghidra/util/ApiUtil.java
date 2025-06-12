/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressOutOfBoundsException;

import ghidra.program.flatapi.FlatProgramAPI;

import ghidra.program.model.data.DataType;
import ghidra.program.model.data.Pointer;
import ghidra.program.model.data.StringDataInstance;

import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Program;

import ghidra.program.model.mem.MemoryBufferImpl;

import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.Varnode;

import java.io.IOException;

/* Anything that isn't directly using JsonWriter to serialize stuff, 
 * especially if it works through the FlatProgramAPI. Goal is to 
 * declutter PcodeSerializer and probably merge into the main Ghidra script again.
 */
class ApiUtil extends FlatProgramAPI {
    Program program;

    public ApiUtil(Program program) {
        super(program);
        this.program = program;
    }

    Address convertAddressToRamSpace(Address address) throws Exception {
        if (address == null) {
            return null;
        }

        try {
            // Note: Function converts address to ramspace only if it belongs to 
            //       constant space; if address space is not constant, return
            if (!address.getAddressSpace().isConstantSpace()) {
                return address;
            }

            // Get the numeric offset and create a new address in RAM space
            long offset = address.getOffset();
            return program.getAddressFactory().getDefaultAddressSpace().getAddress(offset);

        } catch (AddressOutOfBoundsException e) {
            System.out.println(String.format("Error converting address %s to RAM space: %s",
                address, e.getMessage()));
            return null;
        } catch (Exception e) {
            throw new RuntimeException("Failed converting address to RAM space", e);
        }
    }

    public Data getDataReferencedAsConstant(Varnode node) throws Exception {
        // check if node is null
        if (node == null) {
            return null;
        }

            // Only process constant nodes that aren't nulls (address 0 in constant space)
        if (!node.isConstant() || node.getAddress().equals(program.getAddressFactory().getConstantSpace().getAddress(0))) {
            return null;
        }

        // Ghidra sometime fail to resolve references to Data and show it as const. 
        // Check if it is referencing Data as constant from `ram` addresspace.
        // Convert the constant value to a potential RAM address
        Address ramSpaceAddress = convertAddressToRamSpace(node.getAddress());
        if (ramSpaceAddress == null) {
            return null;
        }
        return super.getDataAt(ramSpaceAddress);
    }

    public String findNullTerminatedString(Address address, Pointer pointer) throws Exception {
        if (!address.getAddressSpace().isConstantSpace()) {
            return null;
        }

        Address ramSpaceAddress = convertAddressToRamSpace(address);
        if (ramSpaceAddress == null) {
            return null;
        }
        MemoryBufferImpl memoryBuffer = new MemoryBufferImpl(program.getMemory(), ramSpaceAddress);
        DataType charDataType = pointer.getDataType();
        StringDataInstance stringDataInstance = StringDataInstance.getStringDataInstance(charDataType, memoryBuffer, charDataType.getDefaultSettings(), -1);
        int detectedLength = stringDataInstance.getStringLength();
        if (detectedLength == -1) {
            return null;
        }
        String value = stringDataInstance.getStringValue();
        return value;
    }

    public Data getListingFromAddressAndType(Varnode node, HighVariable highVariable) throws Exception {
        Listing listing = program.getListing();
		return listing.createData(convertAddressToRamSpace(node.getAddress()), highVariable.getDataType());
    }
}