import os
import platform
import subprocess

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration name
config.name = 'PatchestryTest'

# Set the test format to ShTest (shell-based tests)
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# Set up macOS SDK path if on Darwin
if platform.system() == 'Darwin':
    sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode().strip()
    config.environment['SDKROOT'] = sdk_path

config.python_executable = config.python_executable if config.python_executable else sys.executable

# Define file suffixes for test files
config.suffixes = ['.c', '.cpp', '.json']

# Set the root directory where tests should be executed
config.test_exec_root = os.path.join(config.patchestry_obj_root, 'test')

# Set the root directory for test source files
config.test_source_root = os.path.join(config.patchestry_src_root, 'test')

# Set the directory for Ghidra scripts
config.patchestry_script_dir = os.path.join(config.patchestry_src_root, 'scripts', 'ghidra')
config.patchestry_tools_dir = os.path.join(config.patchestry_obj_root, 'tools')

# Set the directory for the test scripts
config.test_scripts_dir = os.path.join(config.patchestry_src_root, 'test', 'scripts')

# Add the Ghidra scripts directory to the substitutions list
config.substitutions.append(('%PATH%', config.patchestry_script_dir))

if 'BUILD_TYPE' in lit_config.params:
    config.patchestry_build_type = lit_config.params['BUILD_TYPE']
else:
    config.patchestry_build_type = "Debug"

def patchestry_tool_path(tool):
    path = [config.patchestry_tools_dir, tool, config.patchestry_build_type]
    return os.path.join(*path, tool)

config.decompiler_headless_tool = os.path.join(config.patchestry_script_dir, 'decompile-headless.sh')
config.pcode_translate_tool = patchestry_tool_path('pcode-translate')

config.json_strip_comments = os.path.join(config.test_scripts_dir, 'strip-json-comments.sh')

config.patchir_decomp_tool = patchestry_tool_path('patchir-decomp')

config.patchir_opt_tool = patchestry_tool_path('patchir-opt')

config.patchir_cir2llvm_tool = patchestry_tool_path('patchir-cir2llvm')

def get_musl_include_path(arch):
    """Get the musl include path for the given architecture on macOS.
    
    Args:
        arch: Target architecture ('x86_64' or 'aarch64')
    Returns:
        The full path to the musl include directory
    """
    try:
        musl_prefix = subprocess.check_output(['brew', '--prefix', 'musl-cross'], 
                                            stderr=subprocess.PIPE).decode().strip()
        return os.path.join(musl_prefix, 'libexec', f'{arch}-linux-musl', 'include')
    except subprocess.CalledProcessError:
        default_path = '/usr/local/opt/musl-cross/libexec/{arch}-linux-musl/include'
        if not os.path.exists(default_path.format(arch=arch)):
            lit_config.fatal('musl-cross not found. Please run: brew install FiloSottile/musl-cross/musl-cross')
        return default_path.format(arch=arch)

def get_target_flags(arch, system):
    """Get compiler flags for the target architecture and system.
    
    Args:
        arch: Target architecture ('x86_64' or 'aarch64')
        system: Host system ('Darwin' or other)
    """
    if system == 'Darwin':
        # On MacOS, use musl cross compiler
        musl_include = get_musl_include_path(arch)
        return [
            f'-target {arch}-linux-musl',
            f'-isystem {musl_include}'
        ]
    else:
        # On Linux, use gnu
        return [f'-target {arch}-linux-gnu']

def get_compiler_command(arch):
    """Get the full compiler command for the target architecture."""
    return FindTool('clang')

# Define tool substitutions
tools = [
    ToolSubst('%file-check', command=FindTool('FileCheck')),
    ToolSubst('%cc-x86_64', command=get_compiler_command('x86_64'), 
              extra_args=get_target_flags('x86_64', platform.system())),
    ToolSubst('%cc-aarch64', command=get_compiler_command('aarch64'),
              extra_args=get_target_flags('aarch64', platform.system())),
    ToolSubst('%cxx', command=FindTool('clang++')),
    ToolSubst('%host_cc', command=config.host_cc),
    ToolSubst('%host_cxx', command=config.host_cxx),
    ToolSubst('%decompile-headless', command=config.decompiler_headless_tool),
    ToolSubst('%pcode-translate', command=config.pcode_translate_tool),
    ToolSubst('%patchir-decomp', command=config.patchir_decomp_tool),
    ToolSubst('%patchir-opt', command=config.patchir_opt_tool),
    ToolSubst('%patchir-cir2llvm', command=config.patchir_cir2llvm_tool),
    ToolSubst('%strip-json-comments', command=config.json_strip_comments),
]


# Process tool substitutions
for tool in tools:
    llvm_config.add_tool_substitutions([tool])

# Add PATH to substitutions
# Set up CI output folder with default path
ci_output_folder = llvm_config.lit_config.params.get('CI_OUTPUT_FOLDER', '')
if not ci_output_folder:
    ci_output_folder = os.path.join(config.test_exec_root, 'ghidra', 'Output')

# Create CI output directory if it doesn't exist
os.makedirs(ci_output_folder, exist_ok=True)

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%ci_output_folder', f"--ci {ci_output_folder}"))
