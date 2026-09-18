from patchestry_llm.decomp import PrintedFunction
from patchestry_llm.model import function_inventory
from patchestry_llm.prompt import SYSTEM_PROMPT, build_tier1_prompt, instruction_lines

from conftest import FUNCTION_KEY


def test_prompt_lists_everything_the_model_may_touch(program):
    inventory = function_inventory(program, FUNCTION_KEY)
    printed = PrintedFunction(key=FUNCTION_KEY, name="FUN_20000000", symbol="FUN_20000000",
                              text="int FUN_20000000(int param_1) {\n    return param_1 + g_state;\n}\n")
    program["functions"][FUNCTION_KEY]["instructions"] = {
        "ram:20000000": {"text": "add r0,r0,r1", "length": 4, "pcode": ["r0 = INT_ADD r0, r1"]}
    }
    prompt = build_tier1_prompt(program, inventory, printed, {"target": "ARM:LE:32:Cortex", "arch": "ARM"},
                                instructions=instruction_lines(program["functions"][FUNCTION_KEY]))
    assert "Target: ARM:LE:32:Cortex (ARM)" in prompt
    assert "Function key: ram:20000000" in prompt
    assert "return param_1 + g_state;" in prompt
    assert "  0 | param_1 | t_int | int | 4" in prompt
    assert "  l0 | iVar1 | t_int | int | 4" in prompt
    assert "  ram:30000000 | g_state | int" in prompt
    assert "  t_int | int | 4" in prompt
    assert "  ram:20000000 | add r0,r0,r1" in prompt
    assert prompt.rstrip().endswith("Reply with the JSON object only.")
    assert '"display_name"' in SYSTEM_PROMPT and "same byte size" in SYSTEM_PROMPT
