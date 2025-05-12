/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.lang.IllegalArgumentException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import ghidra.app.script.GhidraState;
import ghidra.framework.plugintool.PluginTool;
import java.lang.reflect.Field;

import org.junit.jupiter.api.*;

import com.google.gson.stream.JsonWriter;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;

import ghidra.app.util.importer.AutoImporter;
import ghidra.app.util.importer.MessageLog;
import ghidra.app.util.opinion.LoadResults;

import ghidra.framework.model.Project;

import ghidra.program.model.lang.CompilerSpec;
import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.LanguageID;
import ghidra.program.model.lang.LanguageService;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Program;

import ghidra.program.util.DefaultLanguageService;

import ghidra.test.AbstractGhidraHeadlessIntegrationTest;
import ghidra.test.TestEnv;

import ghidra.util.task.TaskMonitor;
import ghidra.app.script.GhidraScript;

public abstract class BaseTest extends AbstractGhidraHeadlessIntegrationTest {

    protected TestEnv env;
    protected Project project;
    protected Program program;

    protected TaskMonitor fakeMonitor = mock(TaskMonitor.class);
    protected String testFwArch = "ARM:LE:32:Cortex";

    protected Program load(File fw, String arch, Project project) throws Exception {
        LanguageService languageService = DefaultLanguageService.getLanguageService();
        LanguageID langId = new LanguageID(arch);
        Language language = languageService.getLanguage(langId);
        CompilerSpec compilerSpec = language.getDefaultCompilerSpec();
        MessageLog log = new MessageLog();

        // lcs == language and compiler spec
        LoadResults<Program> importResults = AutoImporter.importByLookingForLcs(
                fw,
                project,
                project.getProjectData().getRootFolder().getPathname(),
                language,
                compilerSpec,
                this,
                log,
                fakeMonitor
        );
        return (Program) importResults.getPrimaryDomainObject();
    }

    protected Field findFieldInHierarchy(Class<?> clazz, String fieldName) throws NoSuchFieldException {
        Class<?> current = clazz;
        while (current != null) {
            try {
                return current.getDeclaredField(fieldName);
            } catch (NoSuchFieldException e) {
                current = current.getSuperclass();
            }
        }
        throw new NoSuchFieldException("Field '" + fieldName + "' not found in class hierarchy");
    }

    @BeforeAll
    public void setUp() throws Exception {
        env = new TestEnv(PROJECT_NAME);
        assert(env != null);
        project = env.getProject();
        File bloodlightFirmware = new File(System.getenv("BLOODLIGHT_FW_PATH"));
        program = load(bloodlightFirmware, testFwArch, project);
    }

    @AfterAll
    public void cleanUp() throws Exception {
        if (program != null) {
            program.release(this);
        }
        if (env != null) {
            env.dispose();
        }
    }
}
