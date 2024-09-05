
import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration name
config.name = 'PatchestryTest'

# Set the test format to ShTest (shell-based tests)
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.python_executable = config.python_executable if config.python_executable else sys.executable

# Define file suffixes for test files
config.suffixes = ['.c', '.cpp']

# Set the root directory where tests should be executed
config.test_exec_root = os.path.join(config.patchestry_obj_root, 'test')

# Set the root directory for test source files
config.test_source_root = os.path.join(config.patchestry_src_root, 'test')

# Set the directory for Ghidra scripts
config.patchestry_script_dir = os.path.join(config.patchestry_src_root, 'scripts', 'ghidra')

# Add the Ghidra scripts directory to the substitutions list
config.substitutions.append(('%PATH%', config.patchestry_script_dir))

config.decompiler_headless_tool = os.path.join(config.patchestry_script_dir, 'decompile-headless.sh')

# Define tool substitutions
tools = [
    ToolSubst('%file-check', command=FindTool('FileCheck')),
    ToolSubst('%cc', command=FindTool('clang')),
    ToolSubst('%cxx', command=FindTool('clang++')),
    ToolSubst('%host_cc', command=config.host_cc),
    ToolSubst('%host_cxx', command=config.host_cxx),
    ToolSubst('%decompile-headless', command=config.decompiler_headless_tool)
]

# Process tool substitutions
for tool in tools:
    llvm_config.add_tool_substitutions([tool])

# Add PATH to substitutions
config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%ci_output_folder', f"--ci {llvm_config.lit_config.params.get('CI_OUTPUT_FOLDER', '')}"))
