
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
config.test_format = lit.formats.ShTest()

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

# Define tool substitutions
tools = [
    ToolSubst('%decompile-headless', command='decompile-headless.sh'),
    ToolSubst('%file-check', command=FindTool('FileCheck')),
    ToolSubst('%cc', command=FindTool('clang')),
    ToolSubst('%cxx', command=FindTool('clang++')),
    ToolSubst('%host_cc', command=config.host_cc),
    ToolSubst('%host_cxx', command=config.host_cxx)
]

# Process tool substitutions
for tool in tools:
    if tool.command == 'decompile-headless.sh':
        # Set the full path for the decompile-headless.sh script
        path = [config.patchestry_script_dir]
        tool.command = os.path.join(*path, tool.command)
    # Add tool substitutions to LLVM config
    llvm_config.add_tool_substitutions([tool])

# Add test directory to substitutions
config.substitutions.append(('%test_dir', os.path.join(config.test_source_root, 'ghidra')))

# Add PATH to substitutions
config.substitutions.append(('%PATH%', config.environment['PATH']))