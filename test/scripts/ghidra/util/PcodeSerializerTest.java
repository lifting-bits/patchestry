/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
package scripts.ghidra.util;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;

import com.google.gson.stream.JsonWriter;

import ghidra.app.decompiler.DecompInterface;

import ghidra.util.task.TaskMonitor;

import ghidra.program.model.block.BasicBlockModel;

import ghidra.program.model.listing.Function;

import java.util.List;
import java.util.Collections;

import scripts.ghidra.BaseTest;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PcodeSerializerTest extends BaseTest {

    @Test
    public void testConstructor() {
        JsonWriter fakeWriter = mock(JsonWriter.class);
        DecompInterface fakeDecompInterface = mock(DecompInterface.class);
        List<Function> fns = Collections.emptyList();
        PcodeSerializer serializer = new PcodeSerializer(
                fakeWriter,
                fns,
                "Cortex",
                fakeMonitor,
                program,
                fakeDecompInterface,
                new BasicBlockModel(program)
        );

        assertNotNull(serializer.currentProgram);
    }
}
