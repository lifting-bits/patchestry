#!/usr/bin/env python3
"""Generate FileCheck lines for call-target verification.

Parses P-Code JSON and emits FileCheck DAG lines for all direct call targets
in the specified function. Used to verify that patchir-decomp preserves all
call targets from the original binary.

Usage:
    python3 gen-call-checks.py <pcode.json> [--func <name>] [--prefix <PREFIX>]
"""

import argparse
import json
import re
import sys


def find_target_function(functions, func_name):
    """Find the target function entry by name, or return the first with a body."""
    if func_name:
        for fid, fobj in functions.items():
            if fobj.get("name") == func_name:
                return fid, fobj
        return None, None

    # Default: first function with basic_blocks (has a body)
    for fid, fobj in functions.items():
        if "basic_blocks" in fobj:
            return fid, fobj
    return None, None


def sanitize_to_c_identifier(name):
    """Sanitize display names to CIR-safe C identifiers.

    - Replace non-alnum characters with '_', preserving runs of '_'
      (do not collapse multiple underscores).
    - Strip leading/trailing '_' only when they are *synthetic* (introduced
      by the replacement above).  Original ones are meaningful (e.g. "__errno")
      and are preserved, matching the edge-trim in sanitize_to_c_identifier in
      lib/patchestry/Ghidra/JsonDeserialize.cpp.  (That C++ helper also collapses
      interior synthetic underscore runs; this does not — names with replaced
      interior chars can still differ.)
    - Prepend '_' if the identifier would start with a digit.
    - Apply simple normalization for destructor (~Foo) and operator*
      forms (operator<<, operator new, etc.) before generic cleanup.
    """
    if not isinstance(name, str):
        name = str(name)

    # Handle common C++-style special names before generic sanitization.
    if name.startswith("~") and len(name) > 1:
        # Destructor: ~Foo -> dtor_Foo (then sanitized below).
        name = "dtor_" + name[1:]
    elif name.startswith("operator"):
        # Operators: operatorX / operator X / operator<< -> opX / op X / op<<
        # (the rest will be sanitized char-by-char).
        name = "op" + name[len("operator"):]

    # Generic character-wise sanitization: keep alnum and '_', map others to
    # '_'.  Track which '_' are synthetic (from a replaced non-identifier char)
    # so only those can be trimmed from the edges below.
    chars = []
    synthetic = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            chars.append(ch)
            synthetic.append(False)
        else:
            chars.append("_")
            synthetic.append(True)

    # Trim synthetic edge underscores; keep original ones.
    start = 0
    end = len(chars)
    while start < end and chars[start] == "_" and synthetic[start]:
        start += 1
    while end > start and chars[end - 1] == "_" and synthetic[end - 1]:
        end -= 1
    result = "".join(chars[start:end])
    if not result:
        return "fn"
    if result[0].isdigit():
        result = "_" + result
    return result


def get_cir_symbol(func_obj):
    """Return the CIR symbol name for a function.

    C++ mangled names (_Z prefix): CIR uses the mangled name via asm attr.
    Others with display_name: CIR uses sanitized display_name.
    Others without display_name: CIR uses raw name as-is.
    """
    name = func_obj.get("name", "")
    if name.startswith("_Z"):
        return name
    display = func_obj.get("display_name", "")
    if display:
        return sanitize_to_c_identifier(display)
    return name


# Compiler-inserted stack-protector helpers that FunctionBuilder strips during
# AST construction. They are intentionally absent from the decompiled output,
# so the call-target verifier must not require them.
CANARY_CALLEES = frozenset({"__stack_chk_fail", "___stack_chk_fail"})


def collect_call_targets(func, functions):
    """Collect unique direct callee names and count indirect calls."""
    callees = set()
    indirect_count = 0

    for bb in func.get("basic_blocks", {}).values():
        for op in bb.get("operations", {}).values():
            mnemonic = op.get("mnemonic", "")
            if mnemonic == "CALL":
                target = op.get("target", {})
                target_func_id = target.get("function")
                if target_func_id and target_func_id in functions:
                    symbol = get_cir_symbol(functions[target_func_id])
                    if symbol in CANARY_CALLEES:
                        continue
                    callees.add(symbol)
            elif mnemonic == "CALLIND":
                indirect_count += 1

    return sorted(callees), indirect_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate FileCheck call-target verification lines"
    )
    parser.add_argument("input", help="P-Code JSON file (comments already stripped)")
    parser.add_argument("--func", default=None, help="Target function name")
    parser.add_argument("--prefix", default="CALL-CHECK", help="FileCheck prefix")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    functions = data.get("functions", {})
    fid, func = find_target_function(functions, args.func)

    if func is None:
        name = args.func if args.func else "(first with body)"
        print(f"error: function {name} not found", file=sys.stderr)
        sys.exit(1)

    if "basic_blocks" not in func:
        print(f"error: function {func.get('name', '<unknown>')} has no body", file=sys.stderr)
        sys.exit(1)

    callees, indirect_count = collect_call_targets(func, functions)

    print(f"// Call targets for {func['name']}")
    for callee in callees:
        print(f"// {args.prefix}-DAG: cir.call @{callee}")

    if indirect_count > 0:
        print(f"// NOTE: {indirect_count} indirect call(s) (CALLIND) cannot be statically resolved")


if __name__ == "__main__":
    main()
