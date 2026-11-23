#!/usr/bin/env python3
"""List the functions an LLVM IR module declares and does not define.

These are the functions a KLEE run needs a model for.  For each one the
output has the signature as the IR declares it, the attributes that matter
to a model (noreturn, variadic), every call site with the arguments that
are constants, and, when model libraries are given, which of them already
defines it.

Not listed: LLVM intrinsics (`llvm.*`) and KLEE's own functions (`klee_*`).

The input is textual IR (.ll) or bitcode (.bc).  Bitcode is disassembled
with llvm-dis, looked up in --llvm-bin, then $PATCHESTRY_LLVM_PREFIX, then
PATH.  The same goes for --models files.

CLI: list_externals.py MODULE [--models FILE]... [--llvm-bin DIR] [--output FILE]
Exit 0 on success, 1 when the input cannot be read or parsed, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SKIPPED_PREFIXES = ("llvm.", "klee_")
# Words that can stand before the return type of a declaration or call, or
# after the type of a parameter, and are not part of the type.
ATTRIBUTE_WORDS = {
    "dso_local", "dso_preemptable", "extern_weak", "external", "hidden", "protected", "default",
    "noundef", "zeroext", "signext", "inreg", "noalias", "nonnull", "nocapture", "readonly",
    "readnone", "writeonly", "returned", "immarg", "nofree", "nest", "swiftself", "swifterror",
    "tail", "musttail", "notail", "fastcc", "coldcc", "ccc", "unnamed_addr", "local_unnamed_addr",
    "internal", "private", "weak", "weak_odr", "linkonce", "linkonce_odr", "common",
    "available_externally",
    "captures(none)", "nofpclass", "initializes",
}
ATTRIBUTE_CALL = re.compile(
    r"\b(?:align|dereferenceable|dereferenceable_or_null|sret|byval|byref|inalloca|preallocated|"
    r"elementtype|captures|nofpclass|range|initializes|memory|arm_aapcscc|arm_aapcs_vfpcc)"
    r"(?:\s+\d+|\((?:[^()]|\([^()]*\))*\))?")
NAME = r'@(?:"((?:[^"\\]|\\.)*)"|([A-Za-z$._][\w$.\-]*))'


class ParseError(Exception):
    pass


def read_ir(path: Path, llvm_bin: Path | None) -> str:
    data = path.read_bytes()
    if not data.startswith((b"BC\xc0\xde", b"\xde\xc0\x17\x0b")):
        return data.decode("utf-8", errors="replace")
    tool = None
    for directory in (llvm_bin, os.environ.get("PATCHESTRY_LLVM_PREFIX")):
        if directory and (Path(directory) / "llvm-dis").is_file():
            tool = str(Path(directory) / "llvm-dis")
            break
    tool = tool or shutil.which("llvm-dis")
    if tool is None:
        raise ParseError(f"{path} is bitcode and llvm-dis was not found (use --llvm-bin)")
    done = subprocess.run([tool, str(path), "-o", "-"], capture_output=True)
    if done.returncode:
        raise ParseError(f"llvm-dis {path}: {done.stderr.decode(errors='replace').strip()}")
    return done.stdout.decode("utf-8", errors="replace")


def closing(text: str, start: int) -> int:
    """The index of the parenthesis that closes the one at `start`."""
    depth, position, quoted = 0, start, False
    while position < len(text):
        character = text[position]
        if quoted:
            if character == "\\":
                position += 1
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "([{<":
            depth += 1
        elif character in ")]}>":
            depth -= 1
            if depth == 0:
                return position
        position += 1
    raise ParseError(f"unbalanced parentheses in: {text[start:start + 80]}")


def split_top(text: str) -> list[str]:
    parts, depth, current, quoted = [], 0, "", False
    for character in text:
        if quoted:
            quoted = character != '"'
        elif character == '"':
            quoted = True
        elif character in "([{<":
            depth += 1
        elif character in ")]}>":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += character
    if current.strip():
        parts.append(current.strip())
    return parts


def type_of(text: str) -> tuple[str, list[str]]:
    """Split `noundef zeroext i1` into the type and the attributes around it."""
    attributes = [match.group(0) for match in ATTRIBUTE_CALL.finditer(text)]
    words = ATTRIBUTE_CALL.sub(" ", text).split()
    attributes += [word for word in words if word in ATTRIBUTE_WORDS]
    kept = [word for word in words if word not in ATTRIBUTE_WORDS]
    return " ".join(kept), attributes


def parameter_of(text: str) -> dict:
    """One parameter of a declaration, or one argument of a call."""
    # The value, when there is one, is the last word outside brackets.
    # A fixed address of the image: `ptr inttoptr (i32 441964 to ptr)`.
    fixed = re.fullmatch(r"(.*?)\binttoptr\s*\(\s*i\d+\s+(-?\d+)\s+to\s+ptr\s*\)", text.strip())
    if fixed:
        kind, attributes = type_of(fixed.group(1))
        return {"type": kind, "attributes": attributes, "value": f"address:{fixed.group(2)}"}
    kind, attributes = type_of(text)
    value = None
    pieces = kind.rsplit(None, 1)
    if len(pieces) == 2 and not pieces[1].endswith(("*", "]", "}", ">", ")")) \
            and re.match(r"[%@\-\d]|null\b|undef\b|poison\b|true\b|false\b|zeroinitializer\b",
                         pieces[1]):
        kind, value = pieces
    return {"type": kind, "attributes": attributes, "value": value}


def constant_of(value: str | None):
    """What a model can use of an argument: an integer, null, the name of a
    global, or `address:0x...` for a pointer to a fixed address of the image."""
    if value is None:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if value in ("true", "false"):
        return int(value == "true")
    if value == "null":
        return "null"
    if value.startswith("@"):
        return value
    if value.startswith("address:"):
        # Memory of the original image, which the module does not contain.
        return f"address:{int(value[8:]) & 0xFFFFFFFFFFFFFFFF:#x}"
    return None


def scan(text: str) -> dict:
    groups = {match.group(1): match.group(2).split()
              for match in re.finditer(r"^attributes #(\d+) = \{([^}]*)\}", text, re.MULTILINE)}
    module = {"triple": None, "datalayout": None, "defined": [], "definitions": {},
              "declared": {}, "calls": {},
              "address_taken": set()}
    for key in ("triple", "datalayout"):
        match = re.search(rf'^target {key} = "([^"]*)"', text, re.MULTILINE)
        module[key] = match.group(1) if match else None

    current = None
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.split(" ;", 1)[0] if not line.lstrip().startswith(";") else ""
        if not stripped.strip():
            continue
        head = re.match(r"(declare|define)\b(.*)", stripped)
        if head:
            match = re.search(NAME + r"\(", head.group(2))
            if match is None:
                raise ParseError(f"line {number}: no function name in: {stripped[:100]}")
            name = match.group(1) if match.group(1) is not None else match.group(2)
            end = closing(head.group(2), match.end() - 1)
            returned, return_attributes = type_of(head.group(2)[:match.start()])
            parameters = split_top(head.group(2)[match.end():end])
            variadic = bool(parameters) and parameters[-1] == "..."
            tail = head.group(2)[end + 1:].split()
            attributes = [word for word in tail if not word.startswith("#")]
            for word in tail:
                attributes += groups.get(word[1:], []) if word.startswith("#") else []
            signature = {
                "name": name, "line": number, "returns": returned,
                "return_attributes": return_attributes,
                "parameters": [{k: v for k, v in parameter_of(p).items() if k != "value"}
                               for p in parameters if p != "..."],
                "variadic": variadic,
                "noreturn": "noreturn" in attributes,
                "attributes": sorted(set(attributes) - {"{"}),
            }
            if head.group(1) == "define":
                # A definition with internal linkage is not visible to a linker.
                signature["internal"] = bool(
                    {"internal", "private"} & set(head.group(2)[:match.start()].split()))
                module["defined"].append(name)
                module["definitions"][name] = signature
                current = name
            else:
                module["declared"][name] = signature
            continue
        if stripped.startswith("}"):
            current = None
            continue
        consumed = []
        for match in re.finditer(r"\b(?:call|invoke)\b(.*?)" + NAME + r"\(", stripped):
            name = match.group(2) if match.group(2) is not None else match.group(3)
            end = closing(stripped, match.end() - 1)
            consumed.append((match.start(), end))
            arguments = [parameter_of(part) for part in split_top(stripped[match.end():end])]
            result = re.match(r"\s*(%[\w.\-]+|%\"[^\"]*\")\s*=", stripped)
            module["calls"].setdefault(name, []).append({
                "caller": current, "line": number,
                "result_used": result is not None,
                "arguments": [{"type": a["type"], "constant": constant_of(a["value"])}
                              for a in arguments],
            })
        # A function named outside the callee position has its address taken.
        for match in re.finditer(NAME, stripped):
            if any(start <= match.start() <= end for start, end in consumed) and \
                    stripped[match.end():match.end() + 1] == "(":
                continue
            module["address_taken"].add(
                match.group(1) if match.group(1) is not None else match.group(2))
    return module


def externals(module: dict, models: dict[str, str]) -> list[dict]:
    found = []
    for name, declaration in sorted(module["declared"].items()):
        if name.startswith(SKIPPED_PREFIXES) or name in module["defined"]:
            continue
        calls = module["calls"].get(name, [])
        constants = []
        for index in range(max((len(call["arguments"]) for call in calls), default=0)):
            seen = [call["arguments"][index]["constant"]
                    for call in calls if index < len(call["arguments"])]
            # Listed only when every call passes a constant in this position.
            constants.append(None if any(value is None for value in seen)
                             else sorted(set(seen), key=str))
        found.append({
            **declaration,
            "call_count": len(calls),
            "callers": sorted({call["caller"] for call in calls if call["caller"]}),
            "address_taken": name in module["address_taken"],
            "constant_arguments": constants,
            "calls": calls,
            "modelled_in": models.get(name),
        })
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("module", type=Path, help="LLVM IR (.ll) or bitcode (.bc)")
    parser.add_argument("--models", type=Path, action="append", default=[],
                        help="a model library (.ll or .bc); what it defines is marked as modelled")
    parser.add_argument("--llvm-bin", type=Path, help="directory with llvm-dis")
    parser.add_argument("--output", type=Path, help="write the JSON here instead of stdout")
    args = parser.parse_args(argv)

    try:
        module = scan(read_ir(args.module, args.llvm_bin))
        models = {}
        for path in args.models:
            for name in scan(read_ir(path, args.llvm_bin))["defined"]:
                models.setdefault(name, str(path))
    except (OSError, ParseError) as error:
        print(f"list_externals: {error}", file=sys.stderr)
        return 1

    listed = externals(module, models)
    report = {
        "module": str(args.module),
        "triple": module["triple"],
        "datalayout": module["datalayout"],
        "defined": len(module["defined"]),
        "externals": listed,
        "summary": {
            "externals": len(listed),
            "modelled": sum(1 for item in listed if item["modelled_in"]),
            "to_model": sum(1 for item in listed if not item["modelled_in"]),
            "never_called": sorted(item["name"] for item in listed
                                   if not item["call_count"] and not item["address_taken"]),
            # Not externals, and not modelled: the verifier gives a target
            # intrinsic a symbolic result, which loses what it computes.
            "target_intrinsics": sorted(
                name for name in module["declared"]
                if re.match(r"llvm\.(arm|aarch64|thumb|mips|riscv|ppc|x86)\.", name)),
        },
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
