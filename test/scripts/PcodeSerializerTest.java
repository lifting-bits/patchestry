/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;

import com.google.gson.stream.JsonWriter;

import ghidra.app.decompiler.DecompInterface;

import ghidra.test.AbstractGhidraHeadlessIntegrationTest;

import ghidra.util.task.TaskMonitor;

import ghidra.program.model.block.BasicBlockModel;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Program;

import java.util.List;
import java.util.Collections;

import util.PcodeSerializer;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PcodeSerializerTest extends AbstractGhidraHeadlessIntegrationTest {
    Program program;

    @Test
    public void testConstructor() {
        JsonWriter fakeWriter = mock(JsonWriter.class);
        TaskMonitor fakeMonitor = mock(TaskMonitor.class);
        DecompInterface fakeDecompInterface = mock(DecompInterface.class);
        List<Function> fns = Collections.emptyList();
        PcodeSerializer serializer = new PcodeSerializer(
            fakeWriter,
            fns,
            "ARM",
            "Cortex",
            fakeMonitor,
            program,
            fakeDecompInterface,
            new BasicBlockModel(program)
        );

        assertTrue(false);
    }
}
