import copy
import json
from pathlib import Path

import pytest

from patchestry_llm.decomp import DecompError, find_patchir_decomp

SAMPLE_PROGRAM = {
    "architecture": "ARM",
    "id": "ARM:LE:32:Cortex",
    "format": "Executable and Linking Format (ELF)",
    "functions": {
        "ram:20000000": {
            "name": "FUN_20000000",
            "is_intrinsic": False,
            "type": {
                "return_type": "t_int",
                "is_variadic": False,
                "is_noreturn": False,
                "parameter_types": ["t_int"],
            },
            "basic_blocks": {
                "ram:20000000:entry": {
                    "operations": {
                        "p0": {"mnemonic": "DECLARE_PARAMETER", "name": "param_1", "type": "t_int",
                               "kind": "parameter", "index": 0},
                        "l0": {"mnemonic": "DECLARE_LOCAL", "name": "iVar1", "type": "t_int", "kind": "local"},
                        "br": {"mnemonic": "BRANCH", "target_block": "ram:20000000:0:basic"},
                    },
                    "ordered_operations": ["p0", "l0", "br"],
                },
                "ram:20000000:0:basic": {
                    "operations": {
                        "op0": {"mnemonic": "INT_ADD", "type": "t_int",
                                "output": {"kind": "local", "operation": "l0"},
                                "inputs": [{"type": "t_int", "kind": "parameter", "operation": "p0"},
                                           {"type": "t_int", "kind": "global", "global": "ram:30000000"}]},
                        "ret": {"mnemonic": "RETURN",
                                "inputs": [{"type": "t_int", "kind": "local", "operation": "l0"}]},
                    },
                    "ordered_operations": ["op0", "ret"],
                },
            },
            "entry_block": "ram:20000000:entry",
        }
    },
    "globals": {"ram:30000000": {"name": "g_state", "size": "4", "type": "t_int"}},
    "types": {
        "t_int": {"name": "int", "size": 4, "kind": "integer", "is_signed": True},
        "t_u32": {"name": "unsigned int", "size": 4, "kind": "integer", "is_signed": False},
        "t_u8": {"name": "unsigned char", "size": 1, "kind": "integer", "is_signed": False},
    },
}

FUNCTION_KEY = "ram:20000000"


@pytest.fixture
def program() -> dict:
    return copy.deepcopy(SAMPLE_PROGRAM)


@pytest.fixture(scope="session")
def patchir_decomp() -> Path:
    try:
        return find_patchir_decomp()
    except DecompError:
        pytest.skip("patchir-decomp is not built; set PATCHIR_DECOMP")


@pytest.fixture
def program_file(program, tmp_path) -> Path:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(program))
    return path
