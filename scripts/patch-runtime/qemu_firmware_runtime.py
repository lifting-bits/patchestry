#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from elftools.elf.elffile import ELFFile
from patcherex2 import InsertFunctionPatch, ModifyFunctionPatch, Patcherex
from patcherex2.components.compilers import clang_arm as _patcherex_clang_arm
from patcherex2.targets.elf_arm_bare import ElfArmBare


# Patcherex2's ClangArm.compile() post-processes the assembled Thumb bytes and
# rewrites every `bl <known_symbol>` into `blx`. That is correct when patching
# an ARM-mode binary that wants to call into Thumb-compiled patches: the BLX
# (immediate) flips the encoding bit so the CPU exchanges modes on entry.
# For Cortex-M (Thumb-only) firmware, both the trampoline and the patch are
# Thumb, and the rewrite drops us into ARM on every patch call — the M-profile
# core has no ARM decoder and the firmware faults immediately. Skip the
# rewrite by short-circuiting the post-processor to the freshly compiled bytes.
def _thumb_only_compile(self, code, base=0, symbols=None, extension=".c",
                        extra_compiler_flags=None, is_thumb=False, **kwargs):
    if symbols is None:
        symbols = {}
    if extra_compiler_flags is None:
        extra_compiler_flags = []
    extra_compiler_flags = list(extra_compiler_flags)
    extra_compiler_flags += ["-mthumb"] if is_thumb else ["-mno-thumb"]
    # Skip ClangArm.compile entirely; defer to the base Compiler.compile so the
    # bl→blx (and blx→bl) rewrite never runs.
    return _patcherex_clang_arm.Compiler.compile(
        self, code, base=base, symbols=symbols, extension=extension,
        extra_compiler_flags=extra_compiler_flags, **kwargs,
    )


_patcherex_clang_arm.ClangArm.compile = _thumb_only_compile


@dataclass(frozen=True)
class Case:
    name: str
    function_name: str
    spec_name: str
    expected_lines: tuple[str, ...]


@dataclass(frozen=True)
class SymbolInfo:
    name: str
    addr: int
    size: int
    kind: str


CASES: tuple[Case, ...] = (
    Case(
        name="before",
        function_name="qemu_target_before",
        spec_name="qemu_serial_before_patch.yaml",
        expected_lines=(
            "BOOT",
            "PATCH:before",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "DONE",
        ),
    ),
    Case(
        name="after",
        function_name="qemu_target_after",
        spec_name="qemu_serial_after_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:after",
            "BASE:replace",
            "BASE:contract",
            "DONE",
        ),
    ),
    Case(
        name="replace",
        function_name="qemu_target_replace",
        spec_name="qemu_serial_replace_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:replace",
            "BASE:contract",
            "DONE",
        ),
    ),
    Case(
        name="contract",
        function_name="qemu_target_contract",
        spec_name="qemu_serial_entry_contract.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "CONTRACT:entry",
            "BASE:contract",
            "DONE",
        ),
    ),
)

BASELINE_LINES: tuple[str, ...] = (
    "BOOT",
    "BASE:before",
    "BASE:after",
    "BASE:replace",
    "BASE:contract",
    "DONE",
)

PATCHEREX2_LOAD_OPTIONS = {"rebase_granularity": 0x1000}


def run(
    argv: Iterable[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        check=False,
        capture_output=capture_output,
    )
    if result.returncode != 0:
        command = " ".join(str(arg) for arg in argv)
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        details = "\n".join(part for part in (stdout, stderr) if part)
        raise RuntimeError(f"Command failed: {command}\n{details}".rstrip())
    return result


def normalize_transcript(output: bytes) -> tuple[str, ...]:
    text = output.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return tuple(line for line in text.splitlines() if line)


def run_qemu(qemu_system_arm: Path, kernel: Path) -> tuple[str, ...]:
    # Compare against the firmware's UART output only. QEMU's own diagnostic
    # messages (e.g. "Timer with period zero, disabling" emitted by the
    # lm3s6965evb timer model when the firmware idles after DONE) go to stderr
    # and would otherwise leak into the transcript and break the comparison.
    argv = [
        str(qemu_system_arm),
        "-M",
        "lm3s6965evb",
        "-nographic",
        "-kernel",
        str(kernel),
    ]
    try:
        result = subprocess.run(argv, check=False, capture_output=True, timeout=2.0)
        stdout = result.stdout or b""
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
    return normalize_transcript(stdout)


def load_symbols(path: Path) -> dict[str, SymbolInfo]:
    with path.open("rb") as f:
        elf = ELFFile(f)
        symtab = elf.get_section_by_name(".symtab")
        if symtab is None:
            raise RuntimeError(f"Missing .symtab in {path}")
        symbols: dict[str, SymbolInfo] = {}
        for symbol in symtab.iter_symbols():
            if not symbol.name:
                continue
            kind = symbol["st_info"]["type"]
            addr = int(symbol["st_value"])
            if kind == "STT_FUNC":
                addr &= ~1
            symbols[symbol.name] = SymbolInfo(
                name=symbol.name,
                addr=addr,
                size=int(symbol["st_size"]),
                kind=kind,
            )
        return symbols


def detect_arm_bare_regions(path: Path) -> tuple[int, int, int, int, int]:
    """Return (flash_start, flash_end, ram_start, ram_end, entry_point).

    PT_LOAD segments with PF_X and not PF_W are treated as flash; PF_W and
    not PF_X are treated as RAM. Each range is extended past the highest
    populated address so Patcherex2 has room to place inserted functions
    (1 MiB for flash, 64 KiB for RAM, matching the patche_firmware.py
    convention). Fail loudly if the heuristic returns no regions or the
    entry point is unset — silently substituting STM32-like defaults would
    miscompile firmwares with a different memory map (e.g. lm3s6965evb,
    where flash starts at 0x00000000).
    """
    with path.open("rb") as f:
        elf = ELFFile(f)
        flash_regions: list[tuple[int, int]] = []
        ram_regions: list[tuple[int, int]] = []
        for segment in elf.iter_segments():
            if segment["p_type"] != "PT_LOAD":
                continue
            vaddr = int(segment["p_vaddr"])
            end = vaddr + int(segment["p_memsz"])
            flags = int(segment["p_flags"])
            executable = bool(flags & 0x1)
            writable = bool(flags & 0x2)
            if executable and not writable:
                flash_regions.append((vaddr, end))
            elif writable and not executable:
                ram_regions.append((vaddr, end))
        entry_point = int(elf.header["e_entry"]) & ~1

    if not flash_regions:
        raise RuntimeError(
            f"{path}: no executable-only PT_LOAD segments found; cannot "
            "infer flash region for Patcherex2"
        )
    if not ram_regions:
        raise RuntimeError(
            f"{path}: no writable-only PT_LOAD segments found; cannot "
            "infer RAM region for Patcherex2"
        )
    if entry_point == 0:
        raise RuntimeError(f"{path}: ELF entry point is 0; cannot pick a Patcherex2 insert point")

    flash_start = min(start for start, _ in flash_regions)
    flash_end = max(end for _, end in flash_regions) + 0x100000
    ram_start = min(start for start, _ in ram_regions)
    ram_end = max(end for _, end in ram_regions) + 0x10000
    return flash_start, flash_end, ram_start, ram_end, entry_point


_LL_FUNCTION_PATTERN = re.compile(
    r"define\s+(?:dso_local\s+)?(?:\w+\s+)*@(\w+)\([^)]*\)[^{]*\{(?:[^{}]*|\{[^{}]*\})*\}",
    re.MULTILINE | re.DOTALL,
)
_LL_LINKAGE_KEYWORDS = re.compile(
    r"^(?:private|internal|available_externally|linkonce(?:_odr)?"
    r"|weak(?:_odr)?|common|appending|extern_weak|external)\s+",
)


def extract_functions_from_llvm_ir(content: str) -> dict[str, str]:
    """Split an LLVM IR module into one self-contained snippet per function.

    Each snippet carries the module-level type defs, globals, external
    declarations, attribute groups, and metadata, plus `declare`s for the
    *other* functions defined in the module — so it can be compiled in
    isolation by Patcherex2's clang invocation. Mirrors
    Patcherex2/tools/patche_firmware.py:106–210 since that helper is not
    part of the installed `patcherex2` package.
    """
    type_defs = re.findall(
        r"^%[\w.]+ = type\s+(?:opaque|<?\{[^}]*\}>?).*$",
        content,
        re.MULTILINE,
    )
    global_vars = re.findall(
        r"^@[\w.]+ = (?:private |internal |external |global )?(?:constant |global).*$",
        content,
        re.MULTILINE,
    )
    declarations = re.findall(
        r"^declare.*?@\w+\([^)]*\).*?(?:\n|$)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    attributes = re.findall(r"^attributes #\d+ = \{[^}]+\}", content, re.MULTILINE)
    metadata = re.findall(r"^![\w.]+ = .*$", content, re.MULTILINE)

    local_decls: dict[str, str] = {}
    for match in _LL_FUNCTION_PATTERN.finditer(content):
        signature_match = re.match(
            r"(define\s+(?:dso_local\s+)?(?:\w+\s+)*@\w+\([^)]*\)[^{]*)",
            match.group(0),
        )
        if signature_match is None:
            continue
        body_match = re.match(
            r"define\s+(?:dso_local\s+)?(.*?@\w+\([^)]*\))",
            signature_match.group(1),
        )
        if body_match is None:
            continue
        body = _LL_LINKAGE_KEYWORDS.sub("", body_match.group(1))
        local_decls[match.group(1)] = f"declare {body}"

    suffix = "\n\n" + "\n".join(attributes) + "\n\n" + "\n".join(metadata)

    snippets: dict[str, str] = {}
    for match in _LL_FUNCTION_PATTERN.finditer(content):
        name = match.group(1)
        other_decls = [decl for fn, decl in local_decls.items() if fn != name]
        prefix = (
            "\n".join(type_defs)
            + "\n\n"
            + "\n".join(global_vars)
            + "\n\n"
            + "\n".join(declarations)
            + "\n"
            + "\n".join(other_decls)
            + "\n\n"
        )
        snippets[name] = prefix + match.group(0) + suffix
    return snippets


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def candidate_repo_roots(repo_root: Path) -> tuple[Path, ...]:
    candidates = [repo_root]
    if repo_root.parent.name == ".worktrees":
        candidates.append(repo_root.parent.parent)
    return tuple(candidates)


def repo_tool(repo_root: Path, build_type: str, tool_name: str) -> Path:
    # Try every builds/<preset>/ directory so this works for the local
    # `default` preset and CI's `ci` preset (and any other preset added later)
    # without needing to plumb a preset name from the caller.
    tried: list[Path] = []
    for root in candidate_repo_roots(repo_root):
        builds_dir = root / "builds"
        if not builds_dir.is_dir():
            continue
        # Prefer "default" first (legacy local-dev path), then any other preset.
        preset_dirs = sorted(
            (p for p in builds_dir.iterdir() if p.is_dir()),
            key=lambda p: (p.name != "default", p.name),
        )
        for preset_dir in preset_dirs:
            path = preset_dir / "tools" / tool_name / build_type / tool_name
            tried.append(path)
            if path.exists():
                return path
    searched = "\n  ".join(str(p) for p in tried) or "(no builds/ subdirectories found)"
    raise RuntimeError(
        f"Missing tool {tool_name} under any build tree for {repo_root}. "
        f"Build a preset that produces {build_type} artifacts first.\n"
        f"Searched:\n  {searched}"
    )


def patch_function_in_elf(
    original_elf: Path,
    patched_ll: Path,
    target_function: str,
    rewritten_elf: Path,
) -> None:
    snippets = extract_functions_from_llvm_ir(patched_ll.read_text(encoding="utf-8"))
    if target_function not in snippets:
        raise RuntimeError(
            f"Patched LLVM IR {patched_ll} does not define {target_function}. "
            f"Defines: {sorted(snippets)}"
        )

    flash_start, flash_end, ram_start, ram_end, entry_point = detect_arm_bare_regions(original_elf)

    p = Patcherex(
        str(original_elf),
        target_cls=ElfArmBare,
        target_opts={
            "binary_analyzer": "angr",
            "compiler": "clang19",
            "binfmt_tool": "default",
            "allocation_manager": "default",
        },
        components_opts={
            "binfmt_tool": {
                "flash_start": flash_start,
                "flash_end": flash_end,
                "ram_start": ram_start,
                "ram_end": ram_end,
                # ElfArmBare.finalize() injects a save_context push/pop shim
                # at every insert_point to copy any FLASH-resident "new mapped
                # blocks" into RAM at boot. The shim uses force_insert=True
                # and clobbers the 4 bytes at the insert_point without
                # relocating the displaced instructions, so pointing it at
                # entry_point destroys Reset_Handler's .data-init prologue
                # and the firmware fails to reach main. None of the current
                # patches add RAM-resident blocks, so the shim has nothing to
                # do — pass an empty list to skip it. Re-introduce the entry
                # insert_point only if a patch lives in RAM and needs the
                # FLASH→RAM copy stub, and even then point it past the
                # .data/.bss init in startup.S, not at the reset vector.
                "insert_points": [],
            }
        },
    )

    compile_opts = {"extension": ".ll", "load_options": PATCHEREX2_LOAD_OPTIONS}
    # Cortex-M / ElfArmBare targets only execute Thumb. InsertFunctionPatch
    # defaults is_thumb=False, which makes patcherex2 compile the patch with
    # -mno-thumb (ARM mode); the BLX from the Thumb trampoline then jumps
    # into ARM code and the M-profile core faults on the first instruction.
    # Force Thumb for every inserted helper.
    for name, code in snippets.items():
        if name == target_function:
            continue
        p.patches.append(InsertFunctionPatch(name, code, is_thumb=True, compile_opts=compile_opts))
    p.patches.append(
        ModifyFunctionPatch(target_function, snippets[target_function], compile_opts=compile_opts)
    )
    p.apply_patches()
    p.save_binary(str(rewritten_elf))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--build-type", default="Debug")
    parser.add_argument("--qemu-system-arm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--refresh-ghidra-fixtures", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = (
        args.fixture_dir.resolve()
        if args.fixture_dir is not None
        else (repo_root / "test" / "qemu-firmware-runtime" / "fixtures").resolve()
    )

    firmware_dir = repo_root / "firmwares" / "qemu-serial"
    firmware_elf = firmware_dir / "build" / "qemu-serial.elf"
    if not firmware_elf.exists():
        raise RuntimeError(f"Missing firmware ELF at {firmware_elf}. Run the firmware build first.")

    original_symbols = load_symbols(firmware_elf)

    patchir_decomp = repo_tool(repo_root, args.build_type, "patchir-decomp")
    patchir_transform = repo_tool(repo_root, args.build_type, "patchir-transform")
    patchir_cir2llvm = repo_tool(repo_root, args.build_type, "patchir-cir2llvm")
    patchir_yaml_parser = repo_tool(repo_root, args.build_type, "patchir-yaml-parser")

    baseline_lines = run_qemu(args.qemu_system_arm, firmware_elf)
    write_lines(output_dir / "baseline.log", baseline_lines)
    if baseline_lines != BASELINE_LINES:
        raise RuntimeError(
            f"Baseline transcript mismatch.\nExpected: {BASELINE_LINES}\nObserved: {baseline_lines}"
        )

    summary_rows = ["case\tstatus\tartifact"]

    for case in CASES:
        case_dir = output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)

        json_path = case_dir / f"{case.function_name}.json"
        cir_stem = case_dir / case.function_name
        cir_path = case_dir / f"{case.function_name}.cir"
        patched_cir_path = case_dir / f"{case.function_name}.patched.cir"
        patched_ll_path = case_dir / f"{case.function_name}.patched.ll"
        rewritten_elf_path = case_dir / "rewritten.elf"
        transcript_path = case_dir / "qemu.log"

        fixture_json_path = fixture_dir / f"{case.function_name}.json"
        if args.refresh_ghidra_fixtures:
            decompile_script = repo_root / "scripts" / "ghidra" / "decompile-headless.sh"
            run(
                [
                    "bash",
                    str(decompile_script),
                    "--input",
                    str(firmware_elf),
                    "--function",
                    case.function_name,
                    "--output",
                    str(json_path),
                ]
            )
            fixture_json_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(json_path, fixture_json_path)
        else:
            if not fixture_json_path.exists():
                raise RuntimeError(
                    f"Missing checked-in JSON fixture {fixture_json_path}. "
                    "Either provide fixtures with --fixture-dir or rerun with --refresh-ghidra-fixtures."
                )
            shutil.copyfile(fixture_json_path, json_path)

        run([str(patchir_decomp), "-input", str(json_path), "-emit-cir", "-output", str(cir_stem)])

        transform_dir = repo_root / "test" / "patchir-transform"
        run(
            [str(patchir_yaml_parser), case.spec_name, "--validate"],
            cwd=transform_dir,
        )
        run(
            [
                str(patchir_transform),
                str(cir_path),
                "-spec",
                str(transform_dir / case.spec_name),
                "-o",
                str(patched_cir_path),
            ]
        )
        run(
            [
                str(patchir_cir2llvm),
                "-S",
                str(patched_cir_path),
                "-o",
                str(patched_ll_path),
            ]
        )

        if case.function_name not in original_symbols:
            raise RuntimeError(f"Original ELF is missing function symbol {case.function_name}.")

        patch_function_in_elf(
            firmware_elf,
            patched_ll_path,
            case.function_name,
            rewritten_elf_path,
        )

        transcript_lines = run_qemu(args.qemu_system_arm, rewritten_elf_path)
        write_lines(transcript_path, transcript_lines)
        if transcript_lines != case.expected_lines:
            raise RuntimeError(
                f"Transcript mismatch for {case.name}.\nExpected: {case.expected_lines}\nObserved: {transcript_lines}"
            )

        summary_rows.append(f"{case.name}\tpass\t{rewritten_elf_path}")

    summary_tsv = output_dir / "summary.tsv"
    summary_md = output_dir / "summary.md"
    write_lines(summary_tsv, summary_rows)
    write_lines(
        summary_md,
        [
            "# QEMU Firmware Runtime Summary",
            "",
            f"Baseline: {output_dir / 'baseline.log'}",
            "",
            *[f"- `{case.name}`: pass" for case in CASES],
        ],
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
