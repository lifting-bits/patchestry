/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
package util;

import com.google.gson.stream.JsonWriter;

import java.io.BufferedWriter;

import java.util.List;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Program;

// The FunctionSerializer is a utility class of the PatchestryListFunctions script.
public class FunctionSerializer extends JsonWriter {
    protected Program currentProgram;

    public FunctionSerializer(BufferedWriter writer, Program program) {
        super(writer);

        this.currentProgram = program;
    }

    public JsonWriter serialize(List<Function> functions) throws Exception {
        beginObject();
        name("program").value(currentProgram.getName());
        name("functions").beginArray();
        for (Function function : functions) {
            beginObject();
            name("name").value(function.getName());
            name("address").value(function.getEntryPoint().toString());
            name("is_thunk").value(function.isThunk());
            endObject();
        }
        endArray();
        endObject();

        flush();
        return this;
    }
}