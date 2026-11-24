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
| `fp-unit-newer` | the code is built for a floating-point unit with instructions, registers or precision the image's unit does not have, or the two units are not known to be compatible |
| `cpu-profile-mismatch` | the two sides are built for different architecture profiles (A, R, M) |
| `cpu-arch-newer` | the code is built for a newer CPU architecture than the image |
| `cpu-arch-unknown` | the code's architecture is not known to run on the image's |
| `no-attributes` | the image or the object has no build attributes to compare |

A combination this script has no rule for is rejected, not accepted: an
architecture or floating-point tag value it does not know, an A-profile
object against an R-profile image, an M-profile architecture that is not a
subset of the image's.

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
    8: "v6T2", 9: "v6K", 10: "v7", 11: "v6-M", 12: "v6S-M", 13: "v7E-M", 14: "v8",
    15: "v8-R", 16: "v8-M.baseline", 17: "v8-M.mainline", 21: "v8.1-M.mainline",
}
# Tag_CPU_arch_profile is the profile letter as a character code; 0 is an
# architecture from before profiles (up to v6K).
PROFILE_NAMES = {0: "none", ord("A"): "A", ord("R"): "R", ord("M"): "M", ord("S"): "A or R"}
# M-profile architectures do not form one line: v8-M.baseline runs v6-M code
# and not v7-M code.  For each, the architectures whose code it runs.  v7E-M
# is not listed under v8-M.mainline because its DSP instructions are an
# optional extension there.
M_ARCH_RUNS = {
    11: {11, 12},
    12: {11, 12},
    10: {10, 11, 12},
    13: {13, 10, 11, 12},
    16: {16, 11, 12},
    17: {17, 16, 10, 11, 12},
    21: {21, 17, 16, 10, 11, 12},
}
# Tag_FP_arch: the VFP version and whether the unit has 32 double registers
# (16 otherwise).  A later version runs the code of an earlier one.
FP_ARCH = {
    1: (1, False), 2: (2, False), 3: (3, True), 4: (3, False), 5: (4, True), 6: (4, False),
    7: (5, True), 8: (5, False),
}
FP_ARCH_NAMES = {
    0: "none", 1: "VFPv1", 2: "VFPv2", 3: "VFPv3", 4: "VFPv3-D16", 5: "VFPv4",
    6: "VFPv4-D16", 7: "FP-ARMv8", 8: "FP-ARMv8-D16",
}
# clang -mfpu names: (Tag_FP_arch, single precision only) and, for a unit
# with Advanced SIMD, (Tag_FP_arch, Tag_Advanced_SIMD_arch).
MFPU = {
    (2, False): "vfpv2", (3, False): "vfpv3", (4, False): "vfpv3-d16", (5, False): "vfpv4",
    (6, False): "vfpv4-d16", (6, True): "fpv4-sp-d16", (7, False): "fp-armv8",
    (8, False): "fpv5-d16", (8, True): "fpv5-sp-d16",
}
MFPU_SIMD = {(3, 1): "neon", (5, 2): "neon-vfpv4", (7, 3): "neon-fp-armv8"}
MARCH = {
    2: "armv4t", 3: "armv5t", 4: "armv5te", 5: "armv5tej", 6: "armv6", 7: "armv6kz",
    8: "armv6t2", 9: "armv6k", 10: "armv7-a", 14: "armv8-a",
}
MARCH_R = {10: "armv7-r", 15: "armv8-r"}
MARCH_M = {
    10: "armv7-m", 11: "armv6-m", 12: "armv6s-m", 13: "armv7e-m", 16: "armv8-m.base",
    17: "armv8-m.main", 21: "armv8.1-m.main",
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
        "cpu_profile": int(tags.get("TAG_CPU_ARCH_PROFILE", 0)),
        "fp_arch": int(tags.get("TAG_FP_ARCH", 0)),
        # 1: the unit is used for single precision only.
        "fp_single_only": int(tags.get("TAG_ABI_HARDFP_USE", 0)) == 1,
        "simd_arch": int(tags.get("TAG_ADVANCED_SIMD_ARCH", 0)),
        "vfp_args": int(tags.get("TAG_ABI_VFP_ARGS", 0)),
    }


def is_m_profile(attributes: dict) -> bool:
    """Whether the attributes describe an M-profile (Thumb-only) target."""
    return attributes["cpu_profile"] == ord("M") or attributes["cpu_arch"] in (11, 12, 13, 16, 17, 21)


def describe_cpu(attributes: dict) -> str:
    arch = CPU_ARCH_NAMES.get(attributes["cpu_arch"], f"architecture tag {attributes['cpu_arch']}")
    profile = attributes["cpu_profile"]
    if profile in (ord("A"), ord("R"), ord("M")) and attributes["cpu_arch"] in (10, 14):
        arch += f"-{chr(profile)}"
    return f"ARM {arch}"


def cpu_problems(have: dict, want: dict) -> dict[str, str]:
    """Why code built for the CPU of `want` does not run on the CPU of `have`."""
    sides = f"the code is built for {describe_cpu(want)}, the image for {describe_cpu(have)}"
    if is_m_profile(have) != is_m_profile(want):
        return {"cpu-profile-mismatch": sides}
    if is_m_profile(have):
        runs = M_ARCH_RUNS.get(have["cpu_arch"])
        if runs is not None and want["cpu_arch"] in runs:
            return {}
        if have["cpu_arch"] in M_ARCH_RUNS.get(want["cpu_arch"], ()):
            return {"cpu-arch-newer": sides}
        return {"cpu-arch-unknown": sides}
    # A, R and the architectures from before profiles.  An object with no
    # profile is pre-v7 code, which both A and R run.
    lettered = {ord("A"), ord("R")}
    if (have["cpu_profile"] in lettered and want["cpu_profile"] in lettered
            and have["cpu_profile"] != want["cpu_profile"]):
        return {"cpu-profile-mismatch": sides}
    if have["cpu_arch"] not in CPU_ARCH_ORDER or want["cpu_arch"] not in CPU_ARCH_ORDER:
        return {} if have["cpu_arch"] == want["cpu_arch"] else {"cpu-arch-unknown": sides}
    if CPU_ARCH_ORDER.index(want["cpu_arch"]) > CPU_ARCH_ORDER.index(have["cpu_arch"]):
        return {"cpu-arch-newer": sides}
    return {}


def fp_problems(have: dict, want: dict) -> dict[str, str]:
    """Why code built for the floating-point unit of `want` does not run with
    the unit of `have`."""
    if not want["fp_arch"] and not want["simd_arch"]:
        return {}
    if not have["fp_arch"]:
        return {"fp-unit-assumed":
                "the code is built for a floating-point unit; the image declares none"}

    def name(attributes: dict) -> str:
        text = FP_ARCH_NAMES.get(attributes["fp_arch"], f"FP tag {attributes['fp_arch']}")
        if attributes["fp_single_only"]:
            text += " (single precision only)"
        if attributes["simd_arch"]:
            text += f" with Advanced SIMD (tag {attributes['simd_arch']})"
        return text

    sides = f"the code is built for {name(want)}, the image for {name(have)}"
    wanted, present = FP_ARCH.get(want["fp_arch"]), FP_ARCH.get(have["fp_arch"])
    if want["fp_arch"] and (wanted is None or present is None):
        return {"fp-unit-newer": sides + "; one of the two is not a unit this check knows"}
    if wanted is not None and (wanted[0] > present[0] or (wanted[1] and not present[1])):
        return {"fp-unit-newer": sides}
    if have["fp_single_only"] and not want["fp_single_only"] and want["fp_arch"]:
        return {"fp-unit-newer": sides}
    if want["simd_arch"] > have["simd_arch"]:
        return {"fp-unit-newer": sides}
    return {}


def compiler_flags(attributes: dict | None) -> list[str]:
    """clang flags that build code for the ABI the attributes describe."""
    if attributes is None:
        return []
    flags = []
    if is_m_profile(attributes):
        march = MARCH_M.get(attributes["cpu_arch"])
    elif attributes["cpu_profile"] == ord("R"):
        march = MARCH_R.get(attributes["cpu_arch"])
    else:
        march = MARCH.get(attributes["cpu_arch"])
    if march:
        flags.append(f"-march={march}")
    if attributes["vfp_args"] == 1:
        flags.append("-mfloat-abi=hard")
    elif attributes["fp_arch"]:
        flags.append("-mfloat-abi=softfp")
    else:
        flags.append("-mfloat-abi=soft")
    # Without it the compiler picks the default unit of the architecture,
    # which can be a newer one than the image has.
    fpu = MFPU_SIMD.get((attributes["fp_arch"], attributes["simd_arch"])) or MFPU.get(
        (attributes["fp_arch"], attributes["fp_single_only"]))
    if fpu:
        flags.append(f"-mfpu={fpu}")
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
    problems.update(fp_problems(have, want))
    problems.update(cpu_problems(have, want))
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
