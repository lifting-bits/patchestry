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
                    callees.add(functions[target_func_id]["name"])
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
        print(f"error: function {func['name']} has no body", file=sys.stderr)
        sys.exit(1)

    callees, indirect_count = collect_call_targets(func, functions)

    print(f"// Call targets for {func['name']}")
    for callee in callees:
        print(f"// {args.prefix}-DAG: cir.call @{callee}")

    if indirect_count > 0:
        print(f"// NOTE: {indirect_count} indirect call(s) (CALLIND) cannot be statically resolved")


if __name__ == "__main__":
    main()
