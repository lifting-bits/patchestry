# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
import collections
import json
import sys
from typing import Dict, List, Optional

NEXT_ID = 0


def next_id() -> int:
    global NEXT_ID
    next_val = NEXT_ID
    NEXT_ID += 1
    return next_val


ID = collections.defaultdict(next_id)
EMPTY = {}


def should_render_line(op: Dict) -> bool:
    if "output" in op:
        return True


def should_render(data: Dict) -> bool:
    if "output" in data:
        return True
    mnemonic = data['mnemonic']
    if mnemonic == "DECLARE_PARAMETER":
        return False
    if mnemonic.startswith("DECLARE_"):
        return True
    if "target" in data:
        if "is_noreturn" in data["target"]:
            return data["target"]["is_noreturn"]
    return mnemonic in ("BRANCH", "CBRANCH", "BRANCHIND", "RETURN")


def render_typed_var(var: str, type_key: str, types: Dict[str, Dict]) -> str:
    data = types[type_key]
    match data["kind"]:
        case "pointer":
            return render_typed_var("", data["element_type"], types) + " *" + var
        case "typedef":
            return data["name"] + var
        case "void":
            return "void" + var
        case "integer":
            return data["name"] + var
        case "float":
            return data["name"] + var
        case "boolean":
            return data["name"] + var
        case "enum":
            return "enum " + data["name"] + var
        case "struct":
            return "struct " + data["name"] + var
        case "union":
            return "union " + data["name"] + var
        case "array":
            return render_typed_var(var, data["element_type"], types) + "&#91;" + str(data["num_elements"]) + "&#93;"
        case _:
            return var


def render_output(data: Dict[str, str], operations: Dict[str, Dict], vars: Dict[str, Dict]):
    assert "kind" in data
    if data["kind"] == "global":
        print(var_name(vars, data), end=" = ")
    else:
        assert "operation" in data
        data = operations[data["operation"]]
        assert data["mnemonic"].startswith("DECLARE_")
        print(var_name(vars, data), end=" = ")


def render_input(data: Dict, operations: Dict[str, Dict], functions: Dict[str, Dict], vars: Dict[str, Dict], types: Dict[str, Dict]):
    match data["kind"]:
        case "constant":
            print(data["value"], end='')
        case "temporary":
            source = operations[data["operation"]]
            if "name" in source:
                print(var_name(vars, source), end='')
            else:
                print('(', end='')
                render_op(source, operations, functions, vars, types, True)
                print(')', end='')
        case "local":
            print(var_name(vars, operations[data["operation"]]), end='')
        case "parameter":
            print(var_name(vars, operations[data["operation"]]), end='')
        case "global":
            print(var_name(vars, data), end='')
        case "function":
            print(functions[data["function"]]["name"], end='')
        case _:
            print("?INPUT?", end='')


def var_name(vars: Dict[str, Dict], data: Dict) -> str:
    if data["kind"] == "global":
        return vars[data["global"]]["name"]

    name = data["name"]
    if data["kind"] == "temporary":
        name += "_" + str(ID[data["address"]])
    return name


def render_op(data: Dict, operations: Dict[str, Dict], functions: Dict[str, Dict], vars: Dict[str, Dict], types: Dict[str, Dict], inline=False):
    if not should_render(data):
        if not inline:
            return

    mnemonic = data['mnemonic']
    if mnemonic.startswith("DECLARE_"):
        if inline:
            print(data["name"], end='')
        else:
            decl = render_typed_var(" " + var_name(vars, data), data["type"], types)
            print(f"{decl}<BR />", end='')
        return

    is_call = mnemonic in ("CALL", "CALLIND")
    if "output" in data:
        render_output(data["output"], operations, vars)

    if is_call:
        render_input(data["target"], operations, functions, vars, types)
        print("(")
    else:
        print(mnemonic, end=" ")

    sep = ""

    if "inputs" in data:
        for input in data["inputs"]:
            print("", end=sep)
            render_input(input, operations, functions, vars, types)
            sep = ", "

    if "condition" in data:  # For a CBRANCH
        render_input(data["condition"], operations, functions, vars, types)

    if is_call:
        print(")")

    if not inline:
        print("<BR />", end='')


def render_block(key: int, data: Dict, operations: Dict[str, Dict], functions: Dict[str, Dict], vars: Dict[str, Dict], types: Dict[str, Dict]):
    print(f"b{key} [label=<<TABLE cellpadding=\"0\" cellspacing=\"0\" border=\"1\" align=\"left\"><TR><TD align=\"left\">", end='')
    last_op: Optional[Dict] = None
    for op_key in data["ordered_operations"]:
        last_op = operations[op_key]
        render_op(last_op, operations, functions, vars, types)

    print(f"</TD></TR></TABLE>>];")

    if last_op is None:
        return

    if "target_block" in last_op:
        target_key = ID[last_op["target_block"]]
        print(f"b{key} -> b{target_key};")

    if "not_taken_block" in last_op:
        target_key = ID[last_op["not_taken_block"]]
        print(f"b{key} -> b{target_key} [color=\"red\"];")

    if "taken_block" in last_op:
        target_key = ID[last_op["taken_block"]]
        print(f"b{key} -> b{target_key} [color=\"green\"];")


def render_function(func_key: str, functions: Dict[str, Dict], vars: Dict[str, Dict], types: Dict[str, Dict]):
    data = functions[func_key]
    key = ID[func_key]
    basic_blocks: Dict = data.get("basic_blocks", EMPTY)

    # Local the entry block.
    entry_block_name: str = data.get("entry_block", "")
    entry_block: Dict = basic_blocks.get(entry_block_name, EMPTY)

    # Merge all block operations into one dictionary for convenient lookup.
    operations: Dict[str, Dict] = {}
    for block in basic_blocks.values():
        for op_key, op_data in block["operations"].items():
            operations[op_key] = op_data

    # Extract parameter names from the entry block.
    param_types: List[str] = data["type"]["parameter_types"]
    param_names: List[str] = [""] * len(param_types)

    for op in entry_block.get("operations", EMPTY).values():
        if op["mnemonic"] == "DECLARE_PARAMETER":
            param_names[op["index"]] = op["name"]

    func_name: str = data["name"]

    param_str: str = ", ".join(render_typed_var(n, t, types) for n, t in zip(param_names, param_types))
    print(f"f{key} [label=<<TABLE cellpadding=\"0\" cellspacing=\"0\" border=\"1\"><TR><TD>{func_name}({param_str})</TD></TR><TR><TD>{func_key}</TD></TR></TABLE>>];")

    if not entry_block_name:
        return

    entry_block_key = ID[entry_block_name]
    print(f"f{key} -> b{entry_block_key};")

    for block_key, block in basic_blocks.items():
        render_block(ID[block_key], block, operations, functions, vars, types)


def render_functions(data: Dict):
    print("digraph {")
    print("node [shape=none fontname=Courier];")
    functions: Dict[str, Dict] = data["functions"]
    vars: Dict[str, Dict] = data["globals"]
    for func_key in functions.keys():
        render_function(func_key, functions, vars, data["types"])
    print("}")


if __name__ == "__main__":
    with open(sys.argv[1], "r") as json_file:
        render_functions(json.loads(json_file.read()))
