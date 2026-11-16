#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Check that code to insert was compiled for the ABI of the image it goes into.

An ARM ELF says in its build attributes which CPU architecture its code needs,
whether it uses a floating-point unit, and whether floating-point arguments
travel in VFP registers (the hard-float calling convention) or in core
registers.  Code compiled with other settings links and loads without a
complaint; it fails when it runs: an instruction the CPU does not have, or a
`double` handed over in a register the other side does not read.

Problems:

| Problem | Meaning |
| --- | --- |
| `float-abi-mismatch` | the two sides pass floating-point arguments differently |
| `fp-unit-assumed` | the code is built for a floating-point unit the image does not declare |
| `cpu-arch-newer` | the code is built for a newer CPU architecture than the image |
| `no-attributes` | the image or the object has no build attributes to compare |

The comparison is of what each side declares, not of the instructions used.
Code built for a floating-point unit that never touches one is still
reported: the declaration is what the compiler was allowed to do.

CLI: check_abi.py --elf image --object code.o [--output report.json]
Exit 0 when the object fits the image, 1 when not, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import ARMAttributesSection

# Tag_CPU_arch values in the order the architectures were introduced, for the
# A and R profiles.  A later one runs the code of an earlier one.
CPU_ARCH_ORDER = (0, 1, 2, 3, 4, 5, 6, 9, 7, 8, 10, 14)
CPU_ARCH_NAMES = {
    0: "pre-v4", 1: "v4", 2: "v4T", 3: "v5T", 4: "v5TE", 5: "v5TEJ", 6: "v6", 7: "v6KZ",
    8: "v6T2", 9: "v6K", 10: "v7", 14: "v8",
}
MARCH = {
    2: "armv4t", 3: "armv5t", 4: "armv5te", 5: "armv5tej", 6: "armv6", 7: "armv6kz",
    8: "armv6t2", 9: "armv6k", 10: "armv7-a", 14: "armv8-a",
}


def abi_attributes(path: Path) -> dict | None:
    """The ABI-relevant build attributes of an ARM ELF, or None if it has none."""
    try:
        with path.open("rb") as handle:
            elf = ELFFile(handle)
            if elf["e_machine"] != "EM_ARM":
                return None
            tags = {}
            for section in elf.iter_sections():
                if not isinstance(section, ARMAttributesSection):
                    continue
                for subsection in section.iter_subsections():
                    for part in subsection.iter_subsubsections():
                        for attribute in part.iter_attributes():
                            tags[attribute.tag] = attribute.value
    except ELFError:
        return None
    if not tags:
        return None
    return {
        "cpu_arch": int(tags.get("TAG_CPU_ARCH", 0)),
        "fp_arch": int(tags.get("TAG_FP_ARCH", 0)),
        "vfp_args": int(tags.get("TAG_ABI_VFP_ARGS", 0)),
    }


def compiler_flags(attributes: dict | None) -> list[str]:
    """clang flags that build code for the ABI the attributes describe."""
    if attributes is None:
        return []
    flags = []
    march = MARCH.get(attributes["cpu_arch"])
    if march:
        flags.append(f"-march={march}")
    if attributes["vfp_args"] == 1:
        flags.append("-mfloat-abi=hard")
    elif attributes["fp_arch"]:
        flags.append("-mfloat-abi=softfp")
    else:
        flags.append("-mfloat-abi=soft")
    return flags


def abi_problems(image: Path, obj: Path) -> dict[str, str]:
    """Why code in `obj` does not fit `image`; empty when it does."""
    have, want = abi_attributes(image), abi_attributes(obj)
    if have is None or want is None:
        missing = image if have is None else obj
        return {"no-attributes": f"{missing} has no ARM build attributes"}
    problems = {}
    if have["vfp_args"] != want["vfp_args"]:
        def convention(value: int) -> str:
            return "VFP registers" if value == 1 else "core registers"

        problems["float-abi-mismatch"] = (
            f"the image passes floating-point arguments in {convention(have['vfp_args'])}, "
            f"the code in {convention(want['vfp_args'])}"
        )
    if want["fp_arch"] and not have["fp_arch"]:
        problems["fp-unit-assumed"] = (
            "the code is built for a floating-point unit; the image declares none"
        )

    def rank(value: int) -> int:
        return CPU_ARCH_ORDER.index(value) if value in CPU_ARCH_ORDER else len(CPU_ARCH_ORDER)

    if rank(want["cpu_arch"]) > rank(have["cpu_arch"]):
        problems["cpu-arch-newer"] = (
            f"the code is built for ARM {CPU_ARCH_NAMES.get(want['cpu_arch'], want['cpu_arch'])}, "
            f"the image for ARM {CPU_ARCH_NAMES.get(have['cpu_arch'], have['cpu_arch'])}"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--elf", type=Path, required=True, help="image the code goes into")
    parser.add_argument("--object", type=Path, required=True, help="compiled code to insert")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = {
            "image": abi_attributes(args.elf),
            "object": abi_attributes(args.object),
            "problems": abi_problems(args.elf, args.object),
        }
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if report["problems"]:
        print(f"abi: {', '.join(report['problems'])}", file=sys.stderr)
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
