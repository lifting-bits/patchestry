#!/usr/bin/env python3
"""Compile a model library and check it against the functions it is for.

Takes the JSON of list_externals.py (or classify.py) and the C sources of
the models.  Each source is compiled the way the repository builds its own
models (`lib/patchestry/klee/models/CMakeLists.txt`):

    clang -emit-llvm -c -O0 -g -ffreestanding -target x86_64-unknown-linux-gnu \\
        -include klee_stub.h

plus `-I` for the directory of klee_stub.h, so a model kept elsewhere can
include it by name.  Several sources are joined with llvm-link, because
`patchir-klee-verifier --model-library` takes one file.

Errors (exit 1):

| Check | What is wrong |
| --- | --- |
| `compile` | a source does not compile, or the objects do not link into one module |
| `missing` | an external has no model, is not modelled elsewhere, and is not named with --leave |
| `internal` | the model is `static`, so the linker cannot see it |
| `arity` | the model takes a different number of parameters than the IR declares |
| `type` | a parameter or the return value has a different IR type: an `int` or `long` in the model where the firmware has a 32-bit value shows up here |
| `variadic` | one of the two is variadic and the other is not |
| `target` | the library is not built for x86_64 |
| `allocator` | the library defines malloc, calloc, realloc, free, posix_memalign, valloc or memalign, which KLEE provides and the verifier never stubs |
| `verifier` | patchir-klee-verifier (with --verifier) rejects the library or leaves the target without a body |

Warnings (exit 0):

| Check | What to look at |
| --- | --- |
| `extension` | a value narrower than 32 bits is zero-extended on one side and sign-extended on the other |
| `noreturn` | the IR says the function does not return and the model has a `ret` |
| `undefined` | the model calls a function that neither it nor KLEE defines; the verifier will stub it |
| `unused` | a model for a function the module does not declare |
| `left` | an external named with --leave, which gets the verifier's stub: a symbolic return value and no memory effect |

A struct passed by value is compared by its presence only, since the two
modules name their struct types independently.

clang, llvm-link and llvm-dis are looked up in --llvm-bin, then
$PATCHESTRY_LLVM_PREFIX, then PATH.  klee_stub.h is taken from --klee-stub,
or found as lib/patchestry/klee/models/klee_stub.h in a parent of the
current directory or of this script.

CLI: check_models.py EXTERNALS.json MODEL.c... [--leave NAME]... [--leave-file FILE]
         [--bitcode OUT.bc] [--report OUT.json] [--klee-stub FILE] [--llvm-bin DIR]
         [--verifier PATH --module IR --target-function NAME]
Exit 0 when there are no errors, 1 when there are, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import list_externals  # noqa: E402

TARGET = "x86_64-unknown-linux-gnu"
FLAGS = ["-emit-llvm", "-c", "-O0", "-g", "-ffreestanding", "-target", TARGET]
STUB = Path("lib/patchestry/klee/models/klee_stub.h")
# KLEE handles these itself and patchir-klee-verifier never stubs them
# (kKleeAllocators in tools/patchir-klee-verifier/HarnessGen.cpp).  A model
# for one of them would replace KLEE's tracked allocation.
KLEE_PROVIDED = ("malloc", "calloc", "realloc", "free", "posix_memalign", "valloc", "memalign")


class UsageError(Exception):
    pass


def tool(name: str, llvm_bin: Path | None) -> str:
    for directory in (llvm_bin, os.environ.get("PATCHESTRY_LLVM_PREFIX")):
        if directory and (Path(directory) / name).is_file():
            return str(Path(directory) / name)
    found = shutil.which(name)
    if found is None:
        raise UsageError(f"{name} was not found (use --llvm-bin)")
    return found


def klee_stub(given: Path | None) -> Path:
    if given is not None:
        if not given.is_file():
            raise UsageError(f"{given} is not a file")
        return given
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            if (directory / STUB).is_file():
                return directory / STUB
    raise UsageError(f"{STUB} was not found in a parent directory (use --klee-stub)")


def build(sources: list[Path], stub: Path, llvm_bin: Path | None, bitcode: Path,
          problems: list[dict]) -> bool:
    """Compile and join the sources into `bitcode`.  False when that failed."""
    clang = tool("clang", llvm_bin)
    with tempfile.TemporaryDirectory() as scratch:
        objects = []
        for index, source in enumerate(sources):
            out = Path(scratch) / f"{index}.bc"
            # -I as well, so `#include "klee_stub.h"` resolves for a model that
            # is not in the directory of the header.
            done = subprocess.run([clang, *FLAGS, "-include", str(stub), "-I", str(stub.parent),
                                   "-o", str(out), str(source)], capture_output=True, text=True)
            if done.returncode:
                problems.append({"level": "error", "check": "compile", "source": str(source),
                                 "detail": done.stderr.strip()})
                continue
            if done.stderr.strip():
                problems.append({"level": "warning", "check": "compile", "source": str(source),
                                 "detail": done.stderr.strip()})
            objects.append(out)
        if len(objects) != len(sources):
            return False
        if len(objects) == 1:
            shutil.copyfile(objects[0], bitcode)
            return True
        done = subprocess.run([tool("llvm-link", llvm_bin), *map(str, objects),
                               "-o", str(bitcode)], capture_output=True, text=True)
        if done.returncode:
            problems.append({"level": "error", "check": "compile", "source": "llvm-link",
                             "detail": done.stderr.strip()})
            return False
    return True


def shape(kind: str) -> str:
    """A type for comparison across two modules: a named struct is only `struct`."""
    return re.sub(r'%(?:"[^"]*"|[\w.\-]+)', "struct", kind)


def extension(attributes: list[str]) -> str | None:
    return next((word for word in ("zeroext", "signext") if word in attributes), None)


def compare(external: dict, model: dict, has_return: bool) -> list[dict]:
    name = external["name"]

    def problem(level: str, check: str, detail: str) -> dict:
        return {"level": level, "check": check, "function": name, "detail": detail}

    found = []
    if model.get("internal"):
        found.append(problem("error", "internal", "the model has internal linkage (static)"))
    if external["variadic"] != model["variadic"]:
        found.append(problem("error", "variadic",
                             f"declared {'variadic' if external['variadic'] else 'fixed'}, "
                             f"modelled {'variadic' if model['variadic'] else 'fixed'}"))
    wanted, given = external["parameters"], model["parameters"]
    if len(wanted) != len(given):
        found.append(problem("error", "arity", f"declared with {len(wanted)} parameter(s), "
                                               f"modelled with {len(given)}"))
    for index, (left, right) in enumerate(zip(wanted, given)):
        if shape(left["type"]) != shape(right["type"]):
            found.append(problem("error", "type", f"parameter {index} is {left['type']} in the "
                                                  f"module and {right['type']} in the model"))
        elif extension(left["attributes"]) and extension(right["attributes"]) \
                and extension(left["attributes"]) != extension(right["attributes"]):
            found.append(problem("warning", "extension",
                                 f"parameter {index} ({left['type']}) is "
                                 f"{extension(left['attributes'])} in the module and "
                                 f"{extension(right['attributes'])} in the model"))
    if shape(external["returns"]) != shape(model["returns"]):
        found.append(problem("error", "type", f"returns {external['returns']} in the module "
                                              f"and {model['returns']} in the model"))
    elif extension(external["return_attributes"]) and extension(model["return_attributes"]) \
            and extension(external["return_attributes"]) != extension(model["return_attributes"]):
        found.append(problem("warning", "extension",
                             f"the return value ({external['returns']}) is "
                             f"{extension(external['return_attributes'])} in the module and "
                             f"{extension(model['return_attributes'])} in the model"))
    if external["noreturn"] and has_return:
        found.append(problem("warning", "noreturn",
                             "declared noreturn and the model returns on some path"))
    return found


def returning(text: str) -> set[str]:
    """The functions defined in `text` that contain a `ret`."""
    found, current = set(), None
    for line in text.splitlines():
        if line.startswith("define"):
            match = re.search(list_externals.NAME + r"\(", line)
            current = (match.group(1) or match.group(2)) if match else None
        elif line.startswith("}"):
            current = None
        elif current and re.match(r"\s+ret\b", line):
            found.add(current)
    return found


def verify(verifier: Path, module: Path, target: str, bitcode: Path) -> list[dict]:
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "harness.ll"
        done = subprocess.run([str(verifier), str(module), "--target-function", target,
                               "--model-library", str(bitcode), "-S", "-o", str(out)],
                              capture_output=True, text=True)
        if done.returncode or not out.is_file():
            return [{"level": "error", "check": "verifier", "function": target,
                     "detail": (done.stderr or done.stdout).strip()[-2000:]
                               or f"exit {done.returncode}"}]
        harness = list_externals.scan(out.read_text(encoding="utf-8", errors="replace"))
    if target not in harness["defined"]:
        return [{"level": "error", "check": "verifier", "function": target,
                 "detail": "the harness does not define the target function"}]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("externals", type=Path, help="output of list_externals.py or classify.py")
    parser.add_argument("models", type=Path, nargs="+", help="C sources of the models")
    parser.add_argument("--leave", action="append", default=[], metavar="NAME",
                        help="an external deliberately left to the verifier's stub")
    parser.add_argument("--leave-file", type=Path,
                        help="names to leave, one per line; text after # is a comment")
    parser.add_argument("--bitcode", type=Path, help="keep the compiled library here")
    parser.add_argument("--report", type=Path, help="write the findings as JSON")
    parser.add_argument("--klee-stub", type=Path)
    parser.add_argument("--llvm-bin", type=Path)
    parser.add_argument("--verifier", type=Path, help="patchir-klee-verifier, to try the library")
    parser.add_argument("--module", type=Path, help="the IR the externals come from")
    parser.add_argument("--target-function", help="a function of --module to build a harness for")
    args = parser.parse_args(argv)

    problems: list[dict] = []
    try:
        if args.verifier and not (args.module and args.target_function):
            raise UsageError("--verifier needs --module and --target-function")
        for path in (args.externals, *args.models, *filter(None, (args.leave_file, args.module,
                                                                  args.verifier))):
            if not path.is_file():
                raise UsageError(f"{path} is not a file")
        report = json.loads(args.externals.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or "externals" not in report:
            raise UsageError(f"{args.externals} is not the output of list_externals.py")
        leave = set(args.leave)
        if args.leave_file:
            for line in args.leave_file.read_text(encoding="utf-8").splitlines():
                leave |= set(line.split("#", 1)[0].split())
        stub = klee_stub(args.klee_stub)

        with tempfile.TemporaryDirectory() as scratch:
            bitcode = args.bitcode or Path(scratch) / "models.bc"
            built = build(args.models, stub, args.llvm_bin, bitcode, problems)
            text = list_externals.read_ir(bitcode, args.llvm_bin) if built else ""
            if built and args.verifier:
                problems += verify(args.verifier, args.module, args.target_function, bitcode)
    except UsageError as error:
        print(f"check_models: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, list_externals.ParseError) as error:
        print(f"check_models: {error}", file=sys.stderr)
        return 1

    covered, left, elsewhere, provided = [], [], [], []
    if built:
        library = list_externals.scan(text)
        returns = returning(text)
        if not (library["triple"] or "").startswith("x86_64"):
            problems.append({"level": "error", "check": "target",
                             "detail": f"the library is built for {library['triple']}"})
        wanted = {item["name"]: item for item in report["externals"]}
        for name, external in sorted(wanted.items()):
            model = library["definitions"].get(name)
            if name in KLEE_PROVIDED:
                provided.append(name)
                if model is not None:
                    problems.append({"level": "error", "check": "allocator", "function": name,
                                     "detail": "KLEE provides this function; a model would "
                                               "replace its tracked allocation"})
            elif model is not None:
                covered.append(name)
                problems += compare(external, model, name in returns)
            elif name in leave:
                left.append(name)
                problems.append({"level": "warning", "check": "left", "function": name,
                                 "detail": "no model: the verifier stubs it with a symbolic "
                                           "return value and no memory effect"})
            elif external.get("modelled_in"):
                elsewhere.append(name)
            else:
                problems.append({"level": "error", "check": "missing", "function": name,
                                 "detail": "no model, and not named with --leave"})
        for name in sorted(set(leave) - set(wanted)):
            problems.append({"level": "warning", "check": "left", "function": name,
                             "detail": "named with --leave and not an external of the module"})
        for name, model in sorted(library["definitions"].items()):
            if name not in wanted and not model.get("internal"):
                problems.append({"level": "warning", "check": "unused", "function": name,
                                 "detail": "the module does not declare this function"})
        for name in sorted(library["declared"]):
            if not name.startswith(list_externals.SKIPPED_PREFIXES) \
                    and name not in library["defined"]:
                problems.append({"level": "warning", "check": "undefined", "function": name,
                                 "detail": "called by a model and defined nowhere in the library"})

    errors = [p for p in problems if p["level"] == "error"]
    warnings = [p for p in problems if p["level"] == "warning"]
    result = {"externals": len(report["externals"]), "modelled": covered, "left": left,
              "modelled_elsewhere": elsewhere, "provided_by_klee": provided,
              "errors": len(errors), "warnings": len(warnings),
              "bitcode": str(args.bitcode) if args.bitcode and built else None,
              "verified_with": args.target_function if args.verifier and built else None,
              "problems": problems}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"{'failed' if errors else 'ok'}: {len(covered)} of {len(report['externals'])} "
          f"external(s) modelled, {len(left)} left to the stub, {len(elsewhere)} modelled "
          f"elsewhere, {len(provided)} provided by KLEE; {len(errors)} error(s), "
          f"{len(warnings)} warning(s)")
    for item in errors + warnings:
        where = item.get("function") or item.get("source") or "library"
        print(f"  {item['level']}: {item['check']}: {where}: {item['detail']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
